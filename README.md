# Hybrid Content Boosted Neural Collaborative Filtering for Automatic Playlist Genration

This repository is team **spotif.ai**'s main repository of the solution for the [Spotify-RecSys2018 Challenge](https://recsys-challenge.spotify.com/overview). The proposed system is a hybrid recommender system that employs the content of the playlist and the Collaborative Filtering (CF). More speicifically, we use two different recommenders for the two different recommendation scenarios.  

1. Matrix Factorization (MF) based CF model  
  - If seed tracks are already existing, one can deploy well-developed (any kind of) MF based CF system for the recommendatation.  

2. Content Based Neural Collaborative Filtering (NCF)  
  - For the playlists that do not have any seed tracks, we applied a Recurrent Neural Network (RNN) based Content Boosted Neural Collaborative Filtering (CBNCF) model.
  - This model serves the preference score for each tracks within candidates (dataset) per each playlist, based on the track title text 

## Setup the virtual env

To reproduce the entire result or just try out each sub-steps of this solution, you should first install the virtual environment. For this, we recommend for you to install [pipenv](https://docs.pipenv.org/) that we used for this project. If your system already has `python2.7` and `pip`, you can simply run the code below.

```
$(sudo) pip install pipenv
```

Then, clone this repo by

```
$git clone https://github.com/eldrin/recsys18-spotify-spotif-ai.git
```

After the cloning, you need to get into the directory, to install and fire up the environment.

```
$cd recsys18-spotify-spotif-ai
$pipenv install
```

If your main python version is python3.X, make sure install this repo with python2.X option.

```
$pipenv --python 2.7 install
```

Then it'll automatically install python2.7 version of virtualenv in your system. Note that if your system does not have python2.X, you might need to install it manually or use [`pyenv`](https://docs.pipenv.org/advanced/#automatic-python-installation)  

To get into the virtual environment, you can simply hit the command below within the repo top-directory

```
$pipenv shell
```

Now you're good to go!


## Reproduce the result

With all the dependencies installed correctly, you can now reproduce the result, simply by hitting the command below.

```
$python reproduce.py /where/you/decompress/mpd/dataset/ /path/to/challenge_set.json /path/to/dump/outputs/
```


## TODOs
- [x] ~generate readme~
- [ ] provide a script (or notebook) for quick run of the program
- [ ] finish the `README.md` (with proper explanation for everything)
