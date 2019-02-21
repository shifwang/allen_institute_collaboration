import os
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import urllib, urllib2
import json
import pandas as pd
import zipfile
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import time
from gene_surfaceview import *
#from mayavi import mlab
rma_url = ("http://api.brain-map.org/api/v2/data/query.json?criteria=" +
           "model::SectionDataSet,rma::criteria,[failed$eqFalse]," +
           "plane_of_section[name$eqcoronal],products[name$eq'Mouse%20Brain']," +
           "treatments[name$eqISH],rma::include,genes," +
           "rma::options[only$eq'genes.acronym,id'][num_rows$eqall]")

response = urllib2.urlopen(rma_url).read()
data = json.loads(response)
sections = {v["id"]: v["genes"][0]["acronym"] for v in data["msg"]}

i = 0
batch = len(sections)
gene_array = np.zeros((batch, 67, 41, 58))
for section_id in sections:
    start = time.time()
    gene_array[i%batch,:,:,:] = get_section_image(section_id)
    end = time.time()
    if i%100 == 50:
        print('%d images out of %d, sec: %d'%(i+1, len(sections), int(end - start)))
    if (i+1)%batch == 0:
        np.savez('mouse_brain_3d_%d.npz'%(i//batch),sections=sections, gene_array = gene_array)
    i += 1
np.savez('mouse_brain_3d_%d.npz'%(i//batch),sections=sections, gene_array = gene_array)
print('Done')

	
