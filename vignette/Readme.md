# Readme for quipcell vignette

## Getting the data

### Option 1: Downloading the preprocessed data

Download [quipcell_hlca_vignette_data.tar.gz](https://github.com/Genentech/quipcell/releases/download/v0.2/quipcell_hlca_vignette_data.tar.gz),
then from inside the `vignette/` folder run
```
tar xvf /path/to/quipcell_hlca_vignette_data.tar.gz
```

This will create a subfolder `data/` inside the `vignette/` folder.
Then, proceed to the [vignette](vignette.ipynb).

### Option 2: Manually preprocessing the data

Download the h5ad for the HLCA core dataset from:

https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293

Create a `data/` subfolder here, and move the h5ad in there, renaming
it to `hlca_orig.h5ad`.

```
mkdir -p data/
cd data/

# autogenerated curl command from HLCA website
curl -o local.h5ad ...

mv local.h5ad hlca_orig.h5ad
```

Next, run
```
python preprocess.py
```
Which loads the full HLCA, and then performs some processing to make a
smaller dataset that is easier to work with on a laptop. This step can
take a lot of memory; I was able to run it on my laptop after closing
all other applications, but your mileage may vary.

After preprocessing the data, the dataset is much smaller, and the
jupyter notebook for the [vignette](vignette.ipynb) can be run on a
standard laptop.
