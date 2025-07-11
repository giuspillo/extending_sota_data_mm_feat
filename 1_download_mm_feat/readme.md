## Download Multimodal Data

In this folder, we provide the scripts we run to download the raw multimodal data we used (e.g., movie trailers, book covers, music songs)

First, in each subfolder (representing each dataset) we provide a `download_multimodal_data.ipynb` notebook; these notebooks read the extended mappings we provided, that include the multimodal data raw file links, and download them in the correct format and folder, so to be used in the extraction phase. Some links may be broken due to expired content, as has happened to us for some items (which is why we were unable to provide a raw file for each item for each available mode). 

Such notebooks contain more specific information, to support the reader in downloading the original raw data files.

Once raw files are downloaded, we can extract the multimodal featrues.