import numpy as np

class VanishingPointCalculator:
    """Vanishing point detection and calculation"""
    
    @staticmethod
    def vp_calculation_with_pitch(w, h, pitch, focal_length):
        v = np.array([w / 2, 0.0, 1.0])
        vline = np.array([0.0, 1.0, 0.0])
        
        if pitch == 0:
            v[:] = [0, -1, 0]
            vline[:] = [0, 1, h / 2]
        else:
            v[1] = h / 2 - (focal_length / np.tan(np.deg2rad(pitch)))
            vline[2] = (h / 2 + focal_length * np.tan(np.deg2rad(pitch)))
        
        return v, vline
    
    @staticmethod
    def to_pixel_new(v, focal_length):
        """From vpt_transform.py lines 7-10"""
        x = v[0] / v[2] * focal_length * 256 + 256
        y = -v[1] / v[2] * focal_length * 256 + 256
        return x, y

    @staticmethod
    def order_vpt(vps_2D, w=640.0):
        """From vpt_transform.py lines 13-59"""
        h = w  
        vps_2D_ordered = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        dy = abs(vps_2D[:, 1] - h/2)
        dy_max_id = np.where(np.max(dy) - dy < 1)
        dy_max_id = dy_max_id[0]
        
        if dy_max_id.size == 1:
            v3 = vps_2D[dy_max_id[0], :]
            v3_id = dy_max_id[0]
        else:
            dx1 = abs(vps_2D[dy_max_id[0], 0] - w/2)
            dx2 = abs(vps_2D[dy_max_id[1], 0] - w/2)
            if dx1 < dx2:
                v3 = vps_2D[dy_max_id[0], :]
                v3_id = dy_max_id[0]
            else:
                v3 = vps_2D[dy_max_id[1], :]
                v3_id = dy_max_id[1]
        
        v_order = np.array([0, 1, 2])
        vh_id = np.where(v_order != v3_id)
        vh_id = vh_id[0]
        
        if vps_2D[vh_id[0], 0] > vps_2D[vh_id[1], 0]:
            v1 = vps_2D[vh_id[0], :]
            v2 = vps_2D[vh_id[1], :]
        else:
            v1 = vps_2D[vh_id[1], :]
            v2 = vps_2D[vh_id[0], :]
        
        vps_2D_ordered[0, :] = v1
        vps_2D_ordered[1, :] = v2
        vps_2D_ordered[2, :] = v3
        
        return vps_2D_ordered

    @staticmethod
    def transform_vpt(vpts, fov=120.0, orgimg_width=640.0):
        """From vpt_transform.py lines 62-96"""
        v1 = vpts[0]
        v2 = vpts[1]
        v3 = vpts[2]
        
        f = 1 / np.tan(np.deg2rad(fov/2))
        
        p1 = VanishingPointCalculator.to_pixel_new(v1, f)
        p2 = VanishingPointCalculator.to_pixel_new(v2, f)
        p3 = VanishingPointCalculator.to_pixel_new(v3, f)
        
        p1t = np.multiply(p1, orgimg_width / 512)
        p2t = np.multiply(p2, orgimg_width / 512)
        p3t = np.multiply(p3, orgimg_width / 512)
        vpts_2d_t = np.array([p1t, p2t, p3t])
        
        vpts_2d_ordered = VanishingPointCalculator.order_vpt(vpts_2d_t, w=orgimg_width)
        
        return vpts_2d_ordered