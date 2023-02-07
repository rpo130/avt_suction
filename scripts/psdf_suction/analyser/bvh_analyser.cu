struct Line {
    Eigen::Vector3f v, w;

    __host__ __device__ Line(Eigen::Vector3f v, Eigen::Vector3f w)  :   v(v), w(w.normalized()) { }
};

struct Triangle {
    Eigen::Vector3f v[3], n;

    __host__ __device__ Triangle(const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, const Eigen::Vector3f &v3) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;
        n = (v2-v1).cross(v3-v1).normalized();
    }

    __host__ __device__ Triangle() {
        v[0] = Eigen::Vector3f::Zero();
        v[1] = Eigen::Vector3f::Zero();
        v[2] = Eigen::Vector3f::Zero();
        n = Eigen::Vector3f::Zero();
    }

    __host__ __device__ bool Intersect(const Line &l, Eigen::Vector3f *touch_point) {
        // printf("bug1\n");
        // print_mat(v[0].transpose());
        // print_mat(v[1].transpose());
        // print_mat(v[2].transpose());
        // print_mat(n.transpose());
        //  plane coeff: Ax+By+Cz+D=0, [A B C] = n
        float D = - v[0].dot(n);
        //  ray-plane intersect: v' = v + t * w
        float dir = n.dot(l.w);
        if (abs(dir - 0) < __FLT_EPSILON__) return false; //  parallel
        float t = - (n.dot(l.v) + D) / dir;
        if (t < 0) return false;    //  negative direction
        Eigen::Vector3f v_ = l.v + t * l.w;
        //  in-plane
        // print_mat(v_.transpose());
        // printf("%f, %f, %f\n", 
        //         (v[1]-v[0]).cross(v_-v[0]).dot(n), 
        //         (v[2]-v[1]).cross(v_-v[1]).dot(n), 
        //         (v[0]-v[2]).cross(v_-v[2]).dot(n));
        if ((v[1]-v[0]).cross(v_-v[0]).dot(n) >= -__FLT_EPSILON__ &&
            (v[2]-v[1]).cross(v_-v[1]).dot(n) >= -__FLT_EPSILON__ &&
            (v[0]-v[2]).cross(v_-v[2]).dot(n) >= -__FLT_EPSILON__) {
            if (touch_point != nullptr) *touch_point = v_;
            return true;
        }
        else return false;
    }

};

struct BoundingBox {
    Eigen::Vector3f lb, ub;
    
    __host__ __device__ BoundingBox() : lb(Eigen::Vector3f::Zero()), ub(Eigen::Vector3f::Zero()) { }

    __host__ __device__ BoundingBox(const Eigen::Vector3f &lb, const Eigen::Vector3f &ub) : lb(lb), ub(ub) { }

    __host__ __device__ BoundingBox(const BoundingBox &bb1, const BoundingBox &bb2) {
        for (int i = 0; i < 3; i++) {
            lb[i] = bb1.lb[i] < bb2.lb[i] ? bb1.lb[i] : bb2.lb[i];
            ub[i] = bb1.ub[i] > bb2.ub[i] ? bb1.ub[i] : bb2.ub[i];
        }
    }

    __host__ __device__ bool Intersect(const Line &l) {
        // printf("ray\n");
        // print_mat(l.v.transpose());
        // print_mat(l.w.transpose());
        // printf("box\n");
        // print_mat(lb.transpose());
        // print_mat(ub.transpose());
        float t_l = __FLT_MIN__, t_u = __FLT_MAX__;
        for (int i = 0; i < 3; i++) {
            float w_inv = 1 / l.w[i];
            if (isnan(w_inv) || isinf(w_inv)) {
                if (l.v[i] < lb[i] || l.v[i] > ub[i]) return false;
                else continue;
            }
            float t_l_ = (lb[i] - l.v[i]) * w_inv;
            float t_u_ = (ub[i] - l.v[i]) * w_inv;
            if (l.w[i] > 0) {
                t_l = max(t_l, t_l_);
                t_u = min(t_u, t_u_);
            }
            else {
                t_l = max(t_l, t_u_);
                t_u = min(t_u, t_l_);
            }
            // printf("%f %f\n", t_l, t_u);
        }
        // printf("touch %d\n", t_u >= 0 && t_l <= t_u);
        return t_u >= 0 && t_l <= t_u;
    }
};

__global__ void BuildTriangle(float *point_img, int height, int width,
                                int bvh_leafs_offset,
                                Triangle *surfels,
                                BoundingVolumeNode *bvh) {
    //  point image pixel index
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= height * width) return;
    int row = pid / width;
    int col = pid - row * width;
    if (row == height - 1 || col == width - 1) return;

    //  build triangles
    Eigen::Vector3f &p00 = *(Eigen::Vector3f*)(point_img + 3 * (width * (row)   + (col)));
    Eigen::Vector3f &p10 = *(Eigen::Vector3f*)(point_img + 3 * (width * (row+1) + (col)));
    Eigen::Vector3f &p01 = *(Eigen::Vector3f*)(point_img + 3 * (width * (row)   + (col+1)));
    Eigen::Vector3f &p11 = *(Eigen::Vector3f*)(point_img + 3 * (width * (row+1) + (col+1)));
    Eigen::Vector3f p_m = (p00 + p10 + p01 + p11) * 0.25;
    //  surfel image [ 2*(height - 1) x 2*(width - 1) ]
    int width_s = (width - 1) * 2;
    int row_s = row * 2;
    int col_s = col * 2;
    int sid00 = width_s * (row_s)   + (col_s);  
    int sid10 = width_s * (row_s+1) + (col_s);  
    int sid01 = width_s * (row_s)   + (col_s+1);  
    int sid11 = width_s * (row_s+1) + (col_s+1);  
    surfels[sid00] = Triangle(p_m, p01, p00); //  top
    surfels[sid10] = Triangle(p_m, p00, p10); //  left
    surfels[sid01] = Triangle(p_m, p11, p01); //  right
    surfels[sid11] = Triangle(p_m, p10, p11); //  bottom

    //  build leafs
    bvh[bvh_leafs_offset + sid00] = BoundingVolumeNode(surfels + sid00);
    bvh[bvh_leafs_offset + sid10] = BoundingVolumeNode(surfels + sid10);
    bvh[bvh_leafs_offset + sid01] = BoundingVolumeNode(surfels + sid01);
    bvh[bvh_leafs_offset + sid11] = BoundingVolumeNode(surfels + sid11);

    // if (pid == 0) {
    //     print_mat(bvh[bvh_leafs_offset + sid00].surfel->v[0].transpose());
    //     print_mat(bvh[bvh_leafs_offset + sid00].surfel->v[1].transpose());
    //     print_mat(bvh[bvh_leafs_offset + sid00].surfel->v[2].transpose());
    //     print_mat(bvh[bvh_leafs_offset + sid10].surfel->v[0].transpose());
    //     print_mat(bvh[bvh_leafs_offset + sid10].surfel->v[1].transpose());
    //     print_mat(bvh[bvh_leafs_offset + sid10].surfel->v[2].transpose());
    // }
}

__global__ void BuildBVH(BoundingVolumeNode *bvh, 
                        int bvh_old_offset,
                        int bvh_new_offset_row,
                        int bvh_new_offset_col,
                        int height_old, int width_old,
                        int height_new, int width_new) {
    //  reduced image coordinate (height_new x width_new)
    int pid_new = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid_new >= height_new * width_new) return;
    int row_new = pid_new / width_new;
    int col_new = pid_new - row_new * width_new;

    //  layer offset of corresponding old pixels (height_old x width_old)
    //      which is possible out of border of old layer
    int row_old = row_new * 2; // row_old <= height_old-1, row_old == height_old-1 only happens when height_old is odd
    int col_old = col_new * 2; // col_old <= width_old-1, col_old == width_old-1 only happens when width_old is odd
    int pid_old_00 = width_old * (row_old)    + (col_old);
    int pid_old_10 = width_old * (row_old+1)  + (col_old);
    int pid_old_01 = width_old * (row_old)    + (col_old+1);
    int pid_old_11 = width_old * (row_old+1)  + (col_old+1);
    //  layer offset of row reduced layer pixels (height_new x width_old)
    int pid_new_row_00 = row_new * width_old + col_old;
    int pid_new_row_01 = row_new * width_old + col_old + 1;

    //  row reduce
    //      odd height_old && row_old is border
    if (row_old == height_old - 1) {
        bvh[bvh_new_offset_row + pid_new_row_00] = bvh[bvh_old_offset + pid_old_00];
        //  odd width_old && col_old is not border
        if (col_old < width_old - 1)
            bvh[bvh_new_offset_row + pid_new_row_01] = bvh[bvh_old_offset + pid_old_01];
    }
    else {
        bvh[bvh_new_offset_row + pid_new_row_00] = BoundingVolumeNode(bvh + bvh_old_offset + pid_old_00, 
                                                                    bvh + bvh_old_offset + pid_old_10);
        if (col_old < width_old - 1)
            bvh[bvh_new_offset_row + pid_new_row_01] = BoundingVolumeNode(bvh + bvh_old_offset + pid_old_01, 
                                                                            bvh + bvh_old_offset + pid_old_11);
    }

    //  col reduce
    //      odd width_old && col_old is border
    if (col_old == width_old - 1) {
        bvh[bvh_new_offset_col + pid_new] = bvh[bvh_new_offset_row + pid_new_row_00];
    }
    else {
        bvh[bvh_new_offset_col + pid_new] = BoundingVolumeNode(bvh + bvh_new_offset_row + pid_new_row_00, 
                                                                bvh + bvh_new_offset_row + pid_new_row_01);
    }
}

__global__ void Analyse(float *point_img, float *normal_img,
                        unsigned int length, unsigned int width,
                        CupModel cup_model,
                        float s_c, float s_p, float s_f,
                        float normal_threshold, float spring_threshold,
                        BoundingVolumeNode *bvh,
                        float *graspable_img) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= length * width) return;
    int row = pid / width;
    int col = pid - row * width;
	graspable_img[pid] = 0.0f;

	//	grasp pose check
	Eigen::Vector3f &p = *(Eigen::Vector3f*)(point_img + pid * 3);
	Eigen::Vector3f &n = *(Eigen::Vector3f*)(normal_img + pid * 3);
	// Eigen::Vector3f n = { 0.f, 0.f, 1.f };
	if (n[2] < cos(normal_threshold)) return;

	//	transform cup to grasp pose
	Eigen::Vector3f rot_vec = n.cross(Eigen::Vector3f{ 0.f, 0.f, 1.f });
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_world2cup = Eigen::AngleAxisf(- asin(rot_vec.norm()), rot_vec.normalized()).matrix();
	Eigen::Vector3f t_world2cup = p;
    cup_model.a = R_world2cup * cup_model.a + p;
    for (int i = 0; i < 8; i++) {
        cup_model.m[i] = R_world2cup * cup_model.m[i] + p;
    }
	

    // if (pid != 125 * 250 + 125) return;
	//	major point projection
	Eigen::Vector3f m[8];
	for (int i = 0; i < 8; i++) {
        bool touched = bvh->Intersect(Line(cup_model.m[i] + 0.04 * n, -n), &m[i], nullptr);
        // break;
        // printf("cup point %d touched %d\n", i, touched);
        // print_mat((cup_model.m[i] + 0.04 * n).transpose());
		if (!touched) return;
		// float s_c_ = (cup_model.a - m[i]).norm();
		// if ( s_c_ < (1 - spring_threshold) * s_c || s_c_ > (1 + spring_threshold) * s_c) return;
	}

	//	minor point projection && compute spring length
	Eigen::Vector3f m_ij, m_ij_pre;
	Eigen::Vector3f interpolated_point;
	for (int i = 0; i < 8; i++) {

		//	cone spring
		float s_c_ = (cup_model.a - m[i]).norm();
		if ( s_c_ < (1 - spring_threshold) * s_c || s_c_ > (1 + spring_threshold) * s_c) return;

		//	perimeter spring
		float s_p_ = 0.0f;
		m_ij_pre = m[i];
		for (int j = 1; j < 5; j++) {
			float rate = j / 5.0f;
			interpolated_point = cup_model.m[i] * (1-rate) + cup_model.m[(i+1)%8] * rate;
            bool touched = bvh->Intersect(Line(interpolated_point + 0.04 * n, -n), &m_ij, nullptr);
			if (!touched) return;
			s_p_ += (m_ij - m_ij_pre).norm();
			m_ij_pre = m_ij;
		}
		s_p_ += (m[(i+1)%8] - m_ij_pre).norm();
		if ( s_p_ < (1 - spring_threshold) * s_p || s_p_ > (1 + spring_threshold) * s_p) return;

		//	flexion spring
		float s_f_ = 0.0f;
		m_ij_pre = m[i];
		for (int j = 1; j < 5; j++) {
			float rate = j / 5.0f;
			interpolated_point = cup_model.m[i] * (1-rate) + cup_model.m[(i+2)%8] * rate;
            bool touched = bvh->Intersect(Line(interpolated_point + 0.04 * n, -n), &m_ij, nullptr);
			if (!touched) return;
			s_f_ += (m_ij - m_ij_pre).norm();
			m_ij_pre = m_ij;
		}
		s_f_ += (m[(i+2)%8] - m_ij_pre).norm();
		if ( s_f_ < (1 - spring_threshold) * s_f || s_f_ > (1 + spring_threshold) * s_f) return;
	}

	graspable_img[pid] = 1.0f;
}