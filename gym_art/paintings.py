 #%%
import os
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
from typing import Tuple, Union

import json
import random
import itertools

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

import gym
from gym import spaces

HEIGHT=300
WIDTH=300
N_CHANNELS=3

from pathlib import Path
FILE = Path(__file__).resolve()
#FILE = Path('gym_art/paintings.py').resolve()
BASE_PATH=str(FILE.parents[0])
#print(BASE_PATH)
#BASE_PATH='./'
#BASE_PATH='gym/'

#%%
class PaintingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, state_type="pixels"):
        """state_type can be pixels, object"""
        super(PaintingEnv, self).__init__()

        if state_type not in ['pixels', 'object']:
            raise ValueError(f"state_type must be pixels or object but given {state_type}")

        self.viewer = None
        self.state_type = state_type

        # Left, Right
        self.action_space = spaces.Discrete(2)

        # Left and right painting
        #self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([255,255]),
        #    shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([255,255]), dtype=np.uint8)

        # Get the following data
        # - artist_to_val: dict of artist label with their average value
        # - paintings: list of dicts containing info on each painting
        # - painting_pairs: list of tuples containing the id of each painting pair
        self.artist_to_val, self.paintings, self.painting_pairs = load_data()

        # Pre-load in the states
        self.states = self._load_states()

        # Note: each element of self.painting is
        #
        # name, type (concrete, intermediate, abstract), train (true or false)
        # style (Cli etc), phase (1 or 2), value (0-1), display_value (0-100),
        # path (xyz.jpeg)

        # What is the trial number
        self.n_trials = len(self.painting_pairs)
        self.curr_trial = -1 # Idea is that start with blank screen

        return

    def _load_states(self):
        """
        Returns a dictionary of painting ids and states. States can be the image
        or Inception output depending on the value of `self.state_type`
        """
        # Load all of the paintings
        # Potentially also process depending on the settings
        ims = { pid: load_image(pinfo['path']) for pid, pinfo in self.paintings.items() }

        if self.state_type == 'pixels':
            states = ims
        elif self.state_type == 'object':
            nn_out = inception_top_layer([ im for im in ims.values() ])
            states = { pid : nn_out[i,:] for i,pid in enumerate(self.paintings) }

        return states

    def step(self, action):
        """
        Takes a step in this bandit task. Action is either 0 (left) or 1 (right).

        Returns
        -------
        next_state : tuple of nparrays
            Left and right painting as nparray of 300px by 300px.
            If done is True then next_state is (None,None)
        reward : float
            The reward associated with the action (value of the chosen painting).
            Rewards range from 0-1.
            It will be 0 if the task hasn't started yet.
        done : bool
            End of trials
        info : dict
            Keys are 'left' 'right', and 'trial_type'. Values give the
            information on each painting. 'trial_type' can be when both
            paintings are from the 'same artist' or 'different artist'.
        """

        assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Execute one time step within the environment
        if self.curr_trial == -1:
            reward = 0
        else:
            painting_pair = self.painting_pairs[self.curr_trial]
            pid = painting_pair[action]
            select_painting = self.paintings[pid]
            reward = select_painting['display_value']/100.0

        if self.curr_trial == (self.n_trials - 1):
            done = True
            next_state = (None, None)
            info = {}
        else:
            done = False
            self.curr_trial += 1
            left_id, right_id = self.painting_pairs[self.curr_trial]
            left_painting, right_painting = self.paintings[left_id], self.paintings[right_id]
            next_state = (self.states[left_id], self.states[right_id])
            trial_type = 'same artist' if left_painting['style'] == right_painting['style'] else 'different artist'
            info = {'left': left_painting, 'right': right_painting, 'trial_type': trial_type}

        return next_state, reward, done, info


    def reset(self, full_reset:bool=True):
        """
        Full reset will pick a new set of 3 artists out of 5 and assign new mean
        values to each of them.

        Dropped the seed.

        Returns `step`
        """
        #super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        artist_to_val = None if full_reset else self.artist_to_val
        self.artist_to_val, self.paintings, self.painting_pairs = load_data(artist_to_val) # Load the data
        self.states = self._load_states() # Pre-load in the states
        self.curr_trial = -1
        return self.step(0)

    def render(self, mode='human', show_as="default"):
        """
        This will show the two paintings along with the associated reward values
        if that painting is chosen.

        Parameters
        ----------
        mode : str
            Can be "human" or "rgb_array"
        show_as : str
            Can be "default" or "matplotlib"
        """
        if self.curr_trial == -1:
            return None
        painting_ids = self.painting_pairs[self.curr_trial]
        painting_pairs = [ self.paintings[pid] for pid in painting_ids ]
        img_paths = [ painting['path'] for painting in painting_pairs ]
        rewards = [ painting['display_value']/100.0 for painting in painting_pairs ]

        # Load in the images
        imgs = [ Image.open(fn) for fn in img_paths ]
        resized_imgs = [ resize_image(img, (300,300), color=(255,255,255)) for img in imgs ]

        # Render the environment to the screen
        new_im = Image.new("RGB", (800,600), color=(255,255,255))

        # Paste on the two paintings
        new_im.paste(resized_imgs[0], (75,140))
        new_im.paste(resized_imgs[1], (425,140))

        # Add on the reward values
        font = ImageFont.truetype(os.path.join(BASE_PATH, "Arial.ttf"), 30)
        draw = ImageDraw.Draw(new_im)
        rsize1 = draw.textsize(f"({rewards[0]})", font=font)
        draw.text((75 + int((300-rsize1[0])/2), 450), f"({rewards[0]})", (0,0,0), font=font)
        rsize2 = draw.textsize(f"({rewards[1]})", font=font)
        draw.text((425 + int((300-rsize2[0])/2), 450), f"({rewards[1]})", (0,0,0), font=font)

        if mode == "human":
            if show_as == "default":
                new_im.show()
            elif show_as == "matplotlib":
                plt.imshow(new_im)
            else:
                raise ValueError("show_as can only be default or matplotlib")
        elif mode == "rgb_array":
            return np.array(new_im)

        return


#%% Helpers
def resize_image(im:Image.Image, desired_size:Tuple[int,int],
    color:Union[int,float,Tuple[int,int,int],str,None]=0) -> Image.Image:
    """
    Resizes the image to desired size while maintaining the aspect ratio.
    """
    old_size = im.size  # old_size[0] is in (width, height) format

    # Resize the input image keeping its aspect ratio
    # Get it so the width and or height are same as in desired_size
    ratio = min([ float(desired_size[i])/old_size[i] for i in range(2) ])
    new_size = tuple([int(x*ratio) for x in old_size])

    # Resize
    im2 = im.resize(new_size, Image.ANTIALIAS)

    # Add the resized image to a new image with the desired size
    # adds the resize image so it is in the center of the new image
    new_im = Image.new("RGB", desired_size, color=color)
    new_im.paste(im2, ((desired_size[0]-new_size[0])//2,
                        (desired_size[1]-new_size[1])//2))

    return new_im

def load_image(fn):
    img = Image.open(fn)
    img = resize_image(img, (HEIGHT,WIDTH))
    return np.asarray(img)
    # In experiment, I made everything be 300 by 300
    # So I guess would want white background
    # One question is if I want

#%% Load data
def load_data(artist_to_val=None):
    # Loads the basic info for each stimulus
    # dat is a dictionary of dictionaries
    with open(os.path.join(BASE_PATH, "stim_deets_phase1.json")) as f:
        dat = json.load(f)

    # dat['Cli'][0].keys() => name, type, train, style, phase, value, path

    # Randomly pair all the within-pair combos

    # Select 3 artists at random
    if artist_to_val:
        artist_select = list(artist_to_val.keys())
    else:
        artist_names = ['Cli', 'Dek', 'Mon', 'Roh', 'Rot'];
        random.shuffle(artist_names)
        artist_names.append('Kra')
        artist_select = artist_names[:3]

    # Select subset of the data
    sdat = { k : v for k,v in dat.items() if k in artist_select }

    # Randomize the mean value assigned to each artist
    # When I talk about a gallery, I'm referring to one artists paintings
    # Each artist can have abstract, intermediate, or concrete paintings
    if artist_to_val is None:
        learn_mean_vals = [0.2,0.4,0.6]
        random.shuffle(learn_mean_vals)
        artist_to_val = dict(zip(artist_select, learn_mean_vals))

    # Assign the actual value for each painting
    # (this is the reward if they select that painting in cents)
    for styl,mean_val in artist_to_val.items():
        for i in range(len(dat[styl])):
            new_val = (mean_val + sdat[styl][i]['value'])*100
            sdat[styl][i]['display_value'] = int(new_val)

    # WITHIN
    # Half of the trials are between paintings from the same artist
    # Make every possible pair
    within_pairs = []
    for styl in artist_select:
        for i in range(len(sdat[styl])-1):
            for j in range(i+1,len(sdat[styl])):
                # randomize the side
                opt = [sdat[styl][i]['name'], sdat[styl][j]['name']]
                random.shuffle(opt)
                within_pairs.append(tuple(opt))

    # BETWEEN
    # Have an equal number of trials for paintings between artists
    nreps = int((len(within_pairs)*2)/len(artist_select)/len(dat['Cli']))
    #nreps is 5
    gal_elems = {}
    for styl in artist_select:
        gal_elems[styl] = []
        for ii in range(nreps):
            for i in range(len(sdat[styl])):
                gal_elems[styl].append(sdat[styl][i]['name'])
        random.shuffle(gal_elems[styl])

    # Put half the paintings on one side and the other half on the other side
    item_side1 = [ gal_elems[styl][:15] for styl in artist_select ]
    item_side1 = list(itertools.chain(*item_side1)) # flatten
    item_side2 = [ gal_elems[styl][15:] for styl in np.array(artist_select)[[1,2,0]] ]
    item_side2 = list(itertools.chain(*item_side2))

    between_pairs = []
    for i in range(len(item_side1)):
        opt = [item_side1[i], item_side2[i]]
        random.shuffle(opt) # shuffle which side paintings are on
        between_pairs.append(tuple(opt))

    # Combine the within and between trials
    all_pairs = within_pairs + between_pairs
    random.shuffle(all_pairs) # List of tuples with painting id for each side

    # Can make the painting id to this list of painting ids with their info
    flat_vals = list(itertools.chain(*sdat.values()))
    flat_names = [ vals['name'] for vals in flat_vals ]
    paintings = dict(zip(flat_names, flat_vals))

    # Update the path to have 'stimuli'
    for k in paintings:
        paintings[k]['path'] = os.path.join(BASE_PATH, "stimuli", paintings[k]['path'])

    # paintings[all_pairs[0][0]]

    return artist_to_val, paintings, all_pairs


#%% Object Recognition
def preprocess_for_inception(im):
    img = im.resize((299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def inception_top_layer(ims):
    # Setup model
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)

    # Preprocess
    xs = [ preprocess_for_inception(Image.fromarray(im)) for im in ims ]
    xs = np.vstack(xs)

    # Run model
    nn_out = model.predict(xs)

    # Do Max-Pooling across the 8x8 output
    nn_out = nn_out.reshape((nn_out.shape[0],-1,nn_out.shape[-1]))
    nn_out = nn_out.max(axis=1)

    return nn_out

#%% Extra
def tester():
    load_data(None)

    from PIL import Image, ImageDraw
    im = Image.new("RGB", (300,300), color=(255,255,255))
    font = ImageFont.truetype(f"gym/Arial.ttf", 28)
    draw = ImageDraw.Draw(im)
    draw.text((100, 100), "Yes this works", (0,0,0), font=font)
    draw.textsize("Yes this works", font=font)
