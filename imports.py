
#from file_select import FileBrowser
from PIL import Image
import random

import sys
#sys.path.append('../')
from sedinet_models import *
import panel as pn
pn.extension()

import ipywidgets as widgets


configfile = 'config_9percentiles.json'

import json
# load the user configs
with open('config'+os.sep+configfile) as f:    
  config = json.load(f) 

###===================================================
## user defined variables: proportion of data to use for training (a.k.a. the "train/test split")
base    = int(config["base"]) #minimum number of convolutions in a sedinet convolutional block
csvfile = config["csvfile"] #csvfile containing image names and class values
res_folder = config["res_folder"] #folder containing csv file and that will contain model outputs
name = config["name"] #name prefix for output files
dropout = float(config["dropout"]) 
add_bn = bool(config["add_bn"]) 

vars = [k for k in config.keys() if not np.any([k.startswith('base'), k.startswith('res_folder'), k.startswith('csvfile'), k.startswith('name'), k.startswith('dropout'), k.startswith('add_bn')])]

vars = sorted(vars)

###==================================================

csvfile = os.path.abspath(os.getcwd()+os.sep+res_folder+os.sep+csvfile)


## read the data set in, clean and modify the pathnames so they are absolute
df = pd.read_csv(csvfile)
#df.head()

df['files'] = [k.strip() for k in df['files']]
df['files'] = [os.getcwd()+os.sep+f.replace('\\',os.sep) for f in df['files']]  

models = []
for base in [base-2,base,base+2]:
  weights_path = name+"_base"+str(base)+"_model_checkpoint.hdf5"
  ##==============================================
  ## create a SediNet model to estimate sediment category
  model = make_cont_sedinet(base, vars, add_bn, dropout)
  model.load_weights(os.getcwd()+os.sep+'res'+os.sep+res_folder+os.sep+weights_path)
  models.append(model)
    

def get_image():
    #get a random image
    n = random.choice((67364561,67447491,67449081,59858041,94974311,65614911))
    img = 'https://www.allaboutbirds.org/guide/assets/photo/{n}-1280px.jpg'.format(n=n)
    return img

def get_plot():
    im = Image.open(file_input.value)#.convert('LA')
    im = np.array(im) / 255.0    
    plot = plt.imshow(im)
    return plot

