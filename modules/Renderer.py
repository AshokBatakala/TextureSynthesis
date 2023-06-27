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


import pickle
import numpy as np
from pytorch3d.renderer import Textures
from scipy.spatial.transform import Rotation as SciR
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.lighting import AmbientLights

# --------------------------- Renderer class -----------------------------------

class Renderer(nn.Module):
    def __init__(self,
        image_resolution = 224,
        focal_length = 5000,
        faces_per_pixel = 50,            
        blur_radius = 0.0,                  
        device=None
    ):
        
        """
        setup the renderer. with camera and rasterizer
        
        """
        super(Renderer, self).__init__()
        self.image_resolution = image_resolution
        self.focal_length = focal_length
        self.faces_per_pixel = faces_per_pixel
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        image_size = ((self.image_resolution, self.image_resolution),)
        fcl_screen = (self.focal_length*self.image_resolution/224, )                         
        prp_screen = ((self.image_resolution/2, self.image_resolution/2),)


        cameras = PerspectiveCameras(focal_length = fcl_screen,
                                    principal_point = prp_screen,
                                    in_ndc = False,
                                    image_size = image_size,
                                    # R = self.R,
                                    # T = self.T,           # these can be given to renderer forward function
                                    device = self.device)
    
        raster_settings = RasterizationSettings(
            image_size= self.image_resolution,
            blur_radius=blur_radius,
            faces_per_pixel=self.faces_per_pixel,
        )

        lights = AmbientLights(device=self.device)

        self.renderer = MeshRenderer(
                        rasterizer=MeshRasterizer(
                            cameras=cameras,
                            raster_settings=raster_settings
                            ),
                        shader=SoftPhongShader(
                            device=self.device,
                            cameras=cameras,
                            lights=lights
                            )
                            )
    

    def load_input_params(self, **kwargs):
        """
        some of these don't need gradients.
        """
        self.verts_uvs = nn.Parameter(kwargs['verts_uvs'],requires_grad=False)
        self.faces_uvs = nn.Parameter(kwargs['faces_uvs'],requires_grad=False)
        self.texture_map = nn.Parameter(kwargs['maps']) if kwargs['maps'] is not None else None
        self.verts = nn.Parameter(kwargs['verts'])
        self.faces = nn.Parameter(kwargs['faces'],requires_grad=False)
        self.T = nn.Parameter(kwargs['T'])
        self.R = nn.Parameter(kwargs['R'])


    
    def render(self):
        """
        returns : 
        batch of image of shape (batch_size,image_resolution,image_resolution,4)
        """

        self.to(self.device)

        # tex must be computed everytime it renders. else texture_map will not be updated
        # Textures class is deprecated. change it later
        tex = Textures(verts_uvs=self.verts_uvs,
                    faces_uvs=self.faces_uvs,
                    maps= self.texture_map
                    )
        
        self.mesh = Meshes(verts=self.verts, faces = self.faces,textures=tex)

        images = self.renderer(self.mesh, R = self.R, T = self.T)
        images[...,:3]/= 255.0  # normalize rgbt values but not alpha channel.
        image = torch.flip(images, [2]) # this is specific to SMPLMarket dataset.
        return image 
    

    def show(self,i = 0):
        ''' 
        plots the i th image in the Batch of rendered images
        '''
        image = self.render()[i,...,:3]
        plt.figure(figsize=(3,3))
        plt.imshow(image.cpu().detach().numpy())
        plt.axis('off')
        plt.show()

    
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
    

    # ==============================================
    # it will be later shifted to dataloader utils
    # ==============================================

    @staticmethod
    def load_data(  verts_T_paths,
                    standard_body_path = 'data/meta/standard_body.pkl',
                    tm_paths = None,
                    euler_list = None,
                    change_handedness = True,
                    degrees = True,
                    dtype = torch.float32):


        """ 
        inputs :
        ========

        verts_T_paths : list of paths to verts and T pickle files
        standard_body_path : path to the standard body pickle file
        euler_list : list of euler angles for the camera (in degrees)
        degrees : if True, euler angles are in degrees
        tm_paths : list of paths to the texture maps
        change_handedness : if True, flips the y axis of the verts and T 
                    all are torch tensors

                    
        returns a dict :
        =================
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

        # ---------------------------  from pickle file -----------------------

        # load the data
        with open(standard_body_path, 'rb') as f:
            standard_values = pickle.load(f)
        
        verts_uvs = standard_values['vert_uv'].repeat(batch_size,1,1)
        faces_uvs = standard_values['face_uvs'].repeat(batch_size,1,1)
        faces = standard_values['faces'].repeat(batch_size,1,1)

        # --------------------------- return  ---------------------------------
    
        data = {'verts_uvs':verts_uvs,
                'faces_uvs':faces_uvs,
                'maps': texture_map ,
                'verts':verts,
                'faces':faces,
                'T':T,
                'R':R,
                  }
        
        return data
    