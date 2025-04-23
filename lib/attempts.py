class DeformNetV2(nn.Module):
    def __init__(self, opts):
        super(DeformNetV2, self).__init__()
        if opts.normalization == "none":
            opts.normalization = None
        
        self.opts = opts
        self.n_cat = opts.n_cat
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 128, 1),
            nn.ReLU(),
        )
        self.instance_geometry = PointNetfeat(dim=3, num_points=opts.n_pts, 
                    bottleneck_size=opts.bottleneck_size, local_feat=True)
        self.instance_global = DenseFusion(opts.n_pts)

        self.assignment = nn.Sequential(
            nn.Conv1d(2176, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, opts.nv_prior, 1),
        )
        

        # Cage-based deformation
        cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        path_to_mesh_model = 'assets/{}.obj'.format(opts.select_class)
        verts, faces = load_obj(path_to_mesh_model)
        verts = torch.from_numpy(verts).unsqueeze(0).cuda()
        faces = torch.from_numpy(faces).unsqueeze(0).cuda()
        init_mesh = (verts, faces)
        template_vertices, template_faces = self.create_cage(init_mesh)
        self.init_template(template_vertices, template_faces)

        # keypoint predictor
        self.cate_geometry = PointNetfeat(dim=3, num_points=opts.nv_prior, bottleneck_size=opts.bottleneck_size, local_feat=True)
        self.kpt_encode = Linear(opts.bottleneck_size*2, opts.bottleneck_size, activation="lrelu", normalization=opts.normalization)
        self.kpt_decode = MLPDeformer2(dim=3, bottleneck_size=opts.bottleneck_size, npoint=opts.n_keypoints,
                                residual=opts.d_residual, normalization=opts.normalization)
        

        # influence predictor
        influence_size = opts.n_keypoints * self.template_vertices.shape[2]
        shape_encoder_influence = nn.Sequential(
            PointNetfeat(dim=3, num_points=opts.nv_prior, bottleneck_size=influence_size),
            Linear(influence_size, influence_size, activation="lrelu", normalization=opts.normalization))
        dencoder_influence = nn.Sequential(
                Linear(influence_size, influence_size, activation="lrelu", normalization=opts.normalization),
                Linear(influence_size, influence_size, activation=None, normalization=None))
        self.influence_predictor = nn.Sequential(shape_encoder_influence, dencoder_influence)

    # def init_optimizer(self):
    #     params = [{"params": self.influence_predictor.parameters()}]
    #     self.optimizer = torch.optim.Adam(params, lr=self.opt.lr)
    #     self.optimizer.add_param_group({'params': self.influence_param, 'lr': 10 * self.opt.lr})
    #     params = [{"params": self.keypoint_predictor.parameters()}]
    #     self.keypoint_optimizer = torch.optim.Adam(params, lr=self.opt.lr)


    def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
        """
        pull cage vertices as close to the origin, stop when distance to the shape is bellow the threshold
        """
        for _ in range(iters):
            vector = -cage
            current_distance = torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
            min_distance, _ = torch.min(current_distance, dim=2)
            do_update = min_distance > distance
            cage = cage + step * vector * do_update[:, None]
        return cage

    def init_template(self, template_vertices, template_faces):
        # save template as buffer
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        
        # n_keypoints x number of vertices
        self.influence_param = nn.Parameter(torch.zeros(self.opts.n_keypoints, self.template_vertices.shape[2]), requires_grad=True)

    def create_cage(self, init_mesh):
        # cage (1, N, 3)
        init_cage_V = init_mesh[0]
        init_cage_F = init_mesh[1]
        init_cage_V = self.opts.cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F
    
    def forward(self, points, out_img, choose, cat_id, prior, target_shape=None):
        """
        Args:
            points: bs x n_pts x 3
            img: bs x 3 x H x W
            choose: bs x n_pts
            cat_id: bs
            prior: bs x nv x 3

        Returns:
            assign_mat: bs x n_pts x nv
            inst_shape: bs x nv x 3
            deltas: bs x nv x 3
            log_assign: bs x n_pts x nv, for numerical stability

        """
        if self.n_cat == 1:
            cat_id = torch.zeros_like(cat_id, device=cat_id.device)
        bs, n_pts = points.size()[:2]
        nv = prior.size()[1]
        # instance-specific features
        points = points.permute(0, 2, 1)
        inst_local, inst_global = self.instance_geometry(points)
        di = out_img.size()[1]
        rgb_emb = out_img.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)
        rgb_emb = torch.gather(rgb_emb, 2, choose).contiguous()
        rgb_emb = self.instance_color(rgb_emb)
        inst_fuse = self.instance_global(inst_local, rgb_emb)    # bs x 1792 x 1024
        # category-specific features
        cat_prior = prior.permute(0, 2, 1)
        cate_local, cate_global = self.cate_geometry(cat_prior)    # bs x 256
        # assignment features
        assign_feat = torch.cat([inst_local, inst_fuse, cate_global.unsqueeze(-1).repeat(1, 1, n_pts)], dim=1)
        assign_mat = self.assignment(assign_feat)
        assign_mat = assign_mat.permute(0, 2, 1).contiguous() 
        # kpt deformation
        kpt_emb = torch.cat([inst_global, cate_global], dim=-1)
        kpt_emb = self.kpt_encode(kpt_emb)
        kpt_offset = self.kpt_decode(kpt_emb)
        kpt_offset = torch.clamp(kpt_offset, -1.0, 1.0)
        n_fps = self.opts.n_fps if self.opts.n_fps else 2 * self.opts.n_keypoints
        self.init_keypoints = sample_farthest_points(cat_prior, self.opts.n_keypoints)
        cage = self.template_vertices
        if not self.opts.no_optimize_cage:
            cage = self.optimize_cage(cage, cat_prior)
        
        self.influence = self.influence_param[None]
        self.influence_offset = self.influence_predictor(cat_prior)
        self.influence_offset = rearrange(
            self.influence_offset, 'b (k c) -> b k c', k=self.influence.shape[1], c=self.influence.shape[2])
        self.influence = self.influence + self.influence_offset
        distance = torch.sum((self.init_keypoints[..., None] - cage[:, :, None]) ** 2, dim=1)
        n_influence = int((distance.shape[2] / distance.shape[1]) * self.opts.n_influence_ratio)
        n_influence = max(5, n_influence)
        threshold = torch.topk(distance, n_influence, largest=False)[0][:, :, -1]
        threshold = threshold[..., None]
        keep = distance <= threshold
        influence = self.influence * keep

        base_cage = cage
        cage_offset = torch.sum(kpt_offset[..., None] * influence[:, None], dim=2)
        new_cage = base_cage + cage_offset

        cage = cage.transpose(1, 2)
        new_cage = new_cage.transpose(1, 2)
        deformed_shapes, _, _ = deform_with_MVC(
            cage, new_cage, self.template_faces.expand(bs, -1, -1), prior, verbose=True)
        
        return assign_mat, deformed_shapes, inst_fuse, self.influence_offset