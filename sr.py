import argparse
import utils.parse_json as parse_json
import posixpath
import os

from dask.distributed import Client

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

import geometry.internal_orientation as io

if __name__ == "__main__":

    ## parse configuration as .json (or .jsonc) file
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.jsonc',
                        help='JSON file for configuration')

    # parse configs
    args = parser.parse_args()
    options = parse_json.parse(args)
    
    # Convert to NoneDict, which return None for missing key.
    options = parse_json.dict_to_nonedict(options)
    
    root = options["root"]
    images_root = posixpath.join(root, options["images"]["root"])
    images = options["images"]["names"]
    
    ## initialise Dask Client
    # def set_env():
    #     os.environ["GS_NO_SIGN_REQUEST"] = "YES"
    # set_env()
    # client = Client(n_workers=1, threads_per_worker=4)
    # client.run(set_env)
    # print(client)
    
    for i_image, image in enumerate(images):
        """
        TODO:
        créer un autre fichier ipynb pour l'orientation interne.
        Sauver les paramètres intérieurs, shape, etc. dans un csv à la racine des images.
        Utiliser ce script uniquement pour l'orientation externe (et le tester avant sous forme de notebook).
        """
        
        
        
        # image_path = posixpath.join(images_root, image, image + "_a.tif")
        # print(f"Image {i_image}: {image_path}")
        shx, shy = options["images"]["shapes"][i_image]
        
        FMs_path = posixpath.join(images_root, image, "FMs.csv")
        if os.path.exists(FMs_path):
            FMs = pd.read_csv(
                FMs_path,
                index_col=0,
            )
            
            fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
            fig.suptitle(image)
            axs[0].scatter(FMs.x, FMs.y, color='r')
            axs[0].grid()
            axs[0].set_title('Image coordinates []')
            axs[1].scatter(FMs.xi, FMs.eta, color='b')
            axs[1].grid()
            axs[1].set_title('Fiducial coordinates [mm]')
            plt.show()
            
            res = opt.least_squares(
                io.objective_function,
                x0=[0, 0, 0, 127/shx, 127/shy],
                args=(FMs.loc[:, ("xi", "eta")].to_numpy(), FMs.loc[:, ("x", "y")].to_numpy()),
                method="trf",
                # x_scale="jac",
                max_nfev=1000
            )
            
            print(res)

            if res.success:
                params = res.x

                xc, yc, alpha, delta_eta, delta_xi = params[0], params[1], params[2], params[3], params[4]
                print("----------------------------------------------")
                print("|  xc  |  yc  | alpha | delta_eta | delta_xi |") 
                print("| {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(0., 0., 0., 127/shx, 127/shy)) 
                print("| {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(xc, yc, alpha, delta_eta, delta_xi))
                print("----------------------------------------------")

                
                xi, eta = io.image_to_fiducial_coordinates(FMs.loc[:, "x"].values, FMs.loc[:, "y"].values, xc, yc, alpha, delta_eta, delta_xi)
                FMs_inferred_fiducial_coords = np.array([xi, eta]).T
                
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(FMs.xi, FMs.eta, marker=".", color="b", label='FMs true fiducial coords')
                ax.scatter(FMs_inferred_fiducial_coords[:, 0], FMs_inferred_fiducial_coords[:, 1], marker=".", color="r", label='FMs inferred fiducial coords')
                ax.grid()
                ax.legend()
                ax.set_title("Fiducial coordinates [mm] (non linear optimization)")

    
    
    
    
    
    