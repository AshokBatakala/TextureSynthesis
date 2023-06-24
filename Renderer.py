import os
import sys
import torch
import pytorch3d
import torch.nn as nn
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

# add path for demo utils functions
import sys
import os
sys.path.append(os.path.abspath('')) 


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")



import pickle
from pytorch3d.renderer import Textures
from scipy.spatial.transform import Rotation as SciR
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.lighting import AmbientLights
import numpy as np

# --------------------------- Renderer class -----------------------------------

class Renderer(nn.Module):
    def __init__(self,
        image_resolution = 224,
        focal_length = 5000,
        faces_per_pixel = 50,            
        blur_radius = 0.0,                  
        device='cuda:0'
    ):
        
        """
        setup the renderer. with camera and rasterizer
        
        """
        super(Renderer, self).__init__()
        self.image_resolution = image_resolution
        self.focal_length = focal_length
        self.faces_per_pixel = faces_per_pixel
        self.device = device


        image_size = ((self.image_resolution, self.image_resolution),)
        fcl_screen = (self.focal_length*self.image_resolution/224, )                         
        prp_screen = ((self.image_resolution/2, self.image_resolution/2),)


        cameras = PerspectiveCameras(focal_length = fcl_screen,
                                    principal_point = prp_screen,
                                    in_ndc = False,
                                    image_size = image_size,
                                    # R = self.R,
                                    # T = self.T,           # these can be given to renderer forward function
                                    device = device)
    
        raster_settings = RasterizationSettings(
            image_size= self.image_resolution,
            blur_radius=blur_radius,
            faces_per_pixel=self.faces_per_pixel,
        )

        lights = AmbientLights(device=device)

        self.renderer = MeshRenderer(
                        rasterizer=MeshRasterizer(
                            cameras=cameras,
                            raster_settings=raster_settings
                            ),
                        shader=SoftPhongShader(
                            device=device,
                            cameras=cameras,
                            lights=lights
                            )
                            )

    def load_inputs(self,
                    verts_T_paths,
                    standard_body_path = 'data/meta/standard_body.pkl',
                    tm_paths = None,
                    euler_list = None,
                    change_handedness = True,
                    degrees = True,
                    require_grad = True,
                    dtype = torch.float32):


        """ 
        inputs :
        verts_T_paths : list of paths to verts and T pickle files
        standard_body_path : path to the standard body pickle file
        euler_list : list of euler angles for the camera (in degrees)
        degrees : if True, euler angles are in degrees
        tm_paths : list of paths to the texture maps
        change_handedness : if True, flips the y axis of the verts and T
        require_grad : if True, sets requires grad for all the tensors
 
                    all are torch tensors

        verts   : verts of the mesh (batch_size,6890,3)
        T       : translation vector (batch_size,3)
        R       : rotation matrix (batch_size,3,3)
        texture_map : texture map (batch_size,H,W,3) 
                    note: it's range 0 to 255, not 0 to 1. but float

        """

        # get the batch size and assert 
        batch_size = len(verts_T_paths)
        assert tm_paths is None or len(tm_paths) == batch_size, "number of texture map paths should be equal to batch size"
        assert euler_list is None or len(euler_list) == batch_size, "number of euler angles should be equal to batch size"
        

        #verts,T
        verts, T,_,_ = zip(*[pickle.load(open(path, 'rb')) for path in verts_T_paths])
        verts, T = np.array(verts), np.array(T) # to speed up 
        verts, T = torch.tensor(verts, dtype=dtype), torch.tensor(T, dtype=dtype)
        
        if change_handedness:
            verts[:,:,1] *= -1
            T[:,1] *= -1
        
        # R
        R = [SciR.from_euler('zyx', euler, degrees=degrees).as_matrix() for euler in (euler_list or np.zeros((batch_size,3)))]
        R = torch.from_numpy(np.array(R)).type(dtype)
        
        # texture map
        texture_map = torch.stack([torch.Tensor(np.array(plt.imread(path))) for path in (tm_paths or [])], 0).type(dtype) if tm_paths else None
    
        self.verts = verts
        self.T = T
        self.R = R
        self.texture_map = texture_map
        self.batch_size = batch_size
        self.standard_body_path = standard_body_path


        if require_grad:
            self.verts.requires_grad = self.T.requires_grad = self.R.requires_grad = True
            # self.T.requires_grad = True 
            # self.R.requires_grad = True
            if self.texture_map is not None:
                self.texture_map.requires_grad = True



        # --------------------------- loading data  ------------------------------------

        # load the data
        with open(self.standard_body_path, 'rb') as f:
            standard_values = pickle.load(f)
        
        self.verts_uvs = standard_values['vert_uv'].repeat(self.batch_size,1,1)
        self.faces_uvs = standard_values['face_uvs'].repeat(self.batch_size,1,1)
        self.faces = standard_values['faces'].repeat(self.batch_size,1,1)

        # --------------------------- settings  ---------------------------------

        # ---------------
        # shifted below
        # ---------------
        # # Textures class is deprecated. change it later
        # tex = Textures(verts_uvs=self.verts_uvs,
        #             faces_uvs=self.faces_uvs,
        #             maps= self.texture_map
        #             )
        
        # self.mesh = Meshes(verts=self.verts, faces = self.faces,textures=tex)
        

        # below operation makes them non-leaf.so, avoid it
        # self.R = self.R.to(self.device)
        # self.T = self.T.to(self.device)


    def render(self):
        """
        returns : 
        batch of image of shape (batch_size,image_resolution,image_resolution,4)
        """

        # --------------- 
        # tex must be computed everytime it renders. else texture_map will not be updated
        # ---------------
        
        # Textures class is deprecated. change it later
        tex = Textures(verts_uvs=self.verts_uvs,
                    faces_uvs=self.faces_uvs,
                    maps= self.texture_map
                    )
        
        self.mesh = Meshes(verts=self.verts, faces = self.faces,textures=tex)
        



        self.mesh = self.mesh.to(self.device)
        self.renderer = self.renderer.to(self.device)

        images = self.renderer(self.mesh, R = self.R.to(self.device), T = self.T.to(self.device))
        images[...,:3]/= 255.0  # normalize rgb values but not alpha channel.
        # image = images[0, ..., :]
        # get alpha_max 
        # alpha_max = np.max(image[...,3])
        # image[...,3] = np.where(image[...,3] > alpha_max/2, 1.0, 0.0)
        image = torch.flip(images, [2])
        return image 
    
    @staticmethod
    def binary_mask(rendered_image,threshold = 0.5):
        ''' 
        creates binary mask from the rendered image
        rendered_image : (B,H,W,4)
        returns : (B,H,W)
        '''
        image = rendered_image.cpu().detach().numpy()
        alpha_max = np.max(image[...,3])
        image[...,3] = np.where(image[...,3] > alpha_max * threshold, 1.0, 0.0)
        return image[...,3]
    
    def show(self):
        ''' 
        plots the first image in the Batch of rendered images
        '''
        image = self.render()[0,...,:3]
        plt.imshow(image.cpu().detach().numpy())
        plt.show()
