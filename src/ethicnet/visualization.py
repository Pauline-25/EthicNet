import matplotlib.pyplot as plt
from matplotlib import colors
import mpl_toolkits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ethicnet import config


def plot_training_pictures(X_train,nb_pictures = 9):
	'''Plots nb_pictures pictures of X_train'''
	# plot first few images
	for i in range(nb_pictures):
		# define subplot
		plt.subplot(330 + 1 + i)
		# plot raw pixel data
		plt.imshow(X_train[i])
	# show the figure
	plt.show()

def plot_undefined_ethnicity(df):
	'''Plots the 3 pictures where the ethnicity is undefined in UTKFace'''
	df_wrong_ethnicity = df.loc[(df.ethnicity > str(2) ) & (df.ethnicity < str(3) )]
	f, axarr = plt.subplots(1,3)
	k=0
	for image_name in list(df_wrong_ethnicity.image_name):
		axarr[k].imshow(plt.imread(config.data_dir+image_name))
		k+=1

def plot_layer_from_output(output):
	''' 
	output is an array of different arrays (pictures)
	This function plots the differents arrays
	'''
	# nb_filters = output.shape[-1]
	# nrows,ncols = nb_filters // 8 , 8
	# fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(50,50))
	# for y in range(nrows):
	# 	for z in range(ncols):
	# 		ax[y][z].imshow(output[0,:,:,ncols*y+z])
	

	nb_filters = output.shape[-1]
	nrows,ncols = nb_filters // 8 , 8

	fig, axs = plt.subplots(nrows, ncols,figsize=(50,50))
	plt.subplots_adjust(top=1,bottom=0.7,wspace=0)

	images = []
	for y in range(nrows):
		for z in range(ncols):
			# Generate data with a range that varies from one plot to the next.
			images.append(axs[y, z].imshow(output[0,:,:,ncols*y+z],cmap=plt.cm.gray))
			axs[y, z].label_outer()

	# Find the min and max of all colors for use in setting the color scale.
	vmin = min(image.get_array().min() for image in images)
	vmax = max(image.get_array().max() for image in images)
	norm = colors.Normalize(vmin=vmin, vmax=vmax)
	for im in images:
		im.set_norm(norm)

	plt.rcParams.update({'font.size': 25})
	fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.07)
	plt.rcParams.update({'font.size': 10})

	plt.show()