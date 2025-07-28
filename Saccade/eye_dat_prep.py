from read_eeg_gaze import *
import numpy as np
import os
import glob
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


class prepEyeTracking:
	def __init__(self):
		pass
	
	def plot_time_series(self, tser_batch, _xlabel_, _ylabel_, _title_, img_saveloc):
		fig, ax = plt.subplots(figsize=(14, 12))

		lines = [500, 700, 1370, 1970]
		#frequency is 500 Hz
		t = 2*np.arange(tser_batch.shape[1])
		
		_min = np.nanmin(tser_batch) 
		_max = np.nanmax(tser_batch)
		#for all the examples
		colours = plt.cm.rainbow(np.linspace(0, 1, tser_batch.shape[0]))
		for i in range(tser_batch.shape[0]):
			plt.plot(t, tser_batch[i], linewidth=2, label='Train', color=colours[i])

		plt.xlabel(_xlabel_, size=35)
		plt.ylabel(_ylabel_, size=35)
		plt.title(_title_, size=40)
		
		plt.xticks(fontsize=25)
		plt.yticks(fontsize=25)
		plt.rc('xtick', labelsize=25)
		plt.rc('ytick', labelsize=25)
		ax.tick_params(width=8)
		
		plt.ylim(_min - 0.05*np.abs(_min), _max + 0.05*np.abs(_max))
		
		for i in range(int(len(lines))):
			plt.axvline(x=lines[i], color='k', linewidth=2, linestyle='--')
		
		mid_pt = 0.45*(_min + _max)
		#print the events in the task
		plt.text(20, mid_pt, "Fixation", size=30, rotation='vertical', color='k', fontweight='bold')
		plt.text(520, mid_pt, "Stimulus", size=30, rotation='vertical', color='k', fontweight='bold')
		plt.text(1400, mid_pt, "Saccade Onset", size=30, rotation='vertical', color='k', fontweight='bold')
		plt.text(2000, mid_pt, "Probe Onset", size=30, rotation='vertical', color='k', fontweight='bold')

		#plt.legend(fontsize=25, loc ="lower left")
		plt.tight_layout()
		plt.savefig(img_saveloc)
		plt.close()

	def plot_velocity_hist(self, v, logscale=True, nbins=1000, xlabel_="Velocity in pix/s (in logscale)", y_label="Density", title_="Working memory eye-tracking Velocity distribution", saveloc='./Results/EEGNet_LSTM_WM/DPos/eye_vel.png'):
		#plot the histogram of velocities
		ignore_nan_v = np.ravel(v)
		ignore_nan_v = np.delete(ignore_nan_v, np.where(np.isnan(ignore_nan_v)))
		if logscale:
			plt.hist(np.log(ignore_nan_v+1), bins=nbins, density=True)
		else:
			plt.hist(ignore_nan_v, bins=nbins, density=True)
			
		plt.xlabel(xlabel_, fontsize=16)
		plt.ylabel(y_label, fontsize=16)
		plt.title(title_, fontsize=16)
		plt.savefig(saveloc)
		plt.close()

	def conv_to_pix(self, eye, dist=60):
		#dist from screen
		dist = 60 #cm
		dispSize = 24 #in
		screen_dims = [1920, 1080]
		#diag in pix * k = 24 * 2.54 cm (#1 inch = 2.54 cm  (15/6))
		# p k = c => p = c/k
		k = (dispSize * 2.54) / np.sqrt(screen_dims[0]**2 + screen_dims[1]**2)

		eye = dist*np.tan(eye * np.pi/180.0)/k   #eye in pixel
		return eye

	def rem_nans(self, eye_data):
		where_nan = np.where(np.isnan(eye_data)) 
		for i in range(len(where_nan[0])):
			#interpolating by mean in a 12 ms window
			wind_start = np.max([where_nan[1][i]-3, 0])
			wind_end = np.min([where_nan[1][i]+3, eye_data.shape[1]])
			eye_data[where_nan[0][i], where_nan[1][i], where_nan[2][i]] = np.nanmean(eye_data[where_nan[0][i], wind_start:wind_end, where_nan[2][i]])	 
		return eye_data

	def conv1d_same_withnans_smoothing(self, x, f):
		# this will have the code to convolve the data with nans, and keep the dimensions to be the same
		# at the edges it pads with nans do not consider them 
		#if f is odd makes it perfectly symmetric - nevertheless, even values also work
		#data is of the form (t, ): use it as an helper if there are more dimensions - filters are assumed to be flipped as f(x, f) = 1/sum((1_x_i != nan) * f_i) x * f (i) = 1/sum((1_x_i != nan)*f_i)  (1_x_i != nan)sum k_(i-ws/2 to t+ws/2) x_k * f_k 
		#convolution plus weighting with non-nan values
		ws = f.shape[0]
		filtered_data = np.zeros(x.shape)
		#n_pad_left + n_pad_right = ws - 1
		n_pad_left = int((ws - 1)*0.5)
		n_pad_right = ws - n_pad_left - 1 
		#pad the data with nans
		x = np.concatenate([np.ones(n_pad_left)*np.nan, x, np.ones(n_pad_right)*np.nan])
		
		for i in range(x.shape[0] - ws + 1):
			x_i = x[i:i+ws]
			finite_indicator = np.isfinite(x_i)
			#take the sum of the non-nan elements only
			filtered_data[i] = np.nansum(x_i * f) / np.sum(f * finite_indicator) 	

		return filtered_data

	def gaussian_smoothing_eye(self, eye_data, w=3, sd=1.):
		eye_data_smooth = np.zeros(eye_data.shape)
		#of the dimension - (N, t, 2)
		#this was required previously - but not now - as this convolution handles the nan values
		#eye_data[np.where(np.isnan(eye_data))] = 0.
		gauss_filter = np.arange(w)
		mu = np.mean(gauss_filter)
		#want the whole data to be within 3 std of the mean - previously sd = np.std(filter)
		#symmetric filter - can be used without flipping.
		gauss_filter = np.exp(-1.0*(gauss_filter - mu)**2/(2 * sd**2))
		
		
		
		for i in range(eye_data.shape[0]):
			#does not handle the nans	
			#eye_data_smooth[i, :, 0] = np.convolve(eye_data[i, :, 0], filter_norm, 'same')
			#eye_data_smooth[i, :, 1] = np.convolve(eye_data[i, :, 1], filter_norm, 'same')
			#the above code does
			eye_data_smooth[i, :, 0] = self.conv1d_same_withnans_smoothing(eye_data[i, :, 0], gauss_filter)
			eye_data_smooth[i, :, 1] = self.conv1d_same_withnans_smoothing(eye_data[i, :, 1], gauss_filter)


		return eye_data_smooth
	
	def magnitude_of_vel_from_pos(self, x, dt):
		#this does it for a batch and hence the rolling has to be done for every element (i.e., about axis=1)
		x_tp1 = np.roll(x, -1, axis=1)
		x_tm1 = np.roll(x, 1, axis=1)
		#v_t = (x_t+1 - x_t-1)/2dt
		v = (x_tp1 - x_tm1)/(2*dt)
		#deal with the edges by taking 1 sided derivatives
		v[:, 0, :] = (x_tp1 - x)[:, 0, :]/dt 
		v[:, -1, :] = (x - x_tm1)[:, -1, :]/dt
		#magnitude of velocity
		mag_v = np.sqrt(np.sum(v**2, axis=2))
		#print(mag_v)
		return mag_v
	
	def magnitude_of_acc_from_pos(self, x, dt):
		#this does it for a batch and hence the rolling has to be done for every element (i.e., about axis=1)
		x_tp1 = np.roll(x, -1, axis=1)
		x_tm1 = np.roll(x, 1, axis=1)
		#v_t = (x_t+1 - x_t-1)/2dt
		v = (x_tp1 + x_tm1 - 2*x)/(dt**2)
		#deal with the edges by taking 1 sided derivatives
		v[:, 0, :] = (x_tp1 - x)[:, 0, :]/(dt**2) 
		v[:, -1, :] = (x_tm1 - x)[:, -1, :]/(dt**2)
		#magnitude of velocity
		mag_v = np.sqrt(np.sum(v**2, axis=2))
		#print(mag_v)
		return mag_v
	
	def get_saccade_onset(self, v, t=685, lw=-10, rw=150):
		# for the WM task the saccade takes place at 1370 ms -> (500 Hz) at 685th instant, consider rw ms on right side and lw ms on the left (for noisy display)
		#max is the best solution for each trial as it gives the highest velocity for saccade.
		# mask the rest of v
		v[:, 0:t-lw] = 0.
		v[:, t+rw:] = 0.
		
		saccade_onsets = []
		for i in range(v.shape[0]):
			saccade_onsets.append(np.argmax(v[i]))
		
		return np.array(saccade_onsets, dtype='int')

	def rem_blinks(self, a, threshold=10, n_flicks=5):
		# lost eye-tracking data - removing blinks
		list_to_rem = []
		for i in range(a.shape[0]):
			curr_trial = a[i]
			curr_trial = np.delete(curr_trial, np.where(np.isnan(curr_trial)))

			n_blinks = np.sum(np.log(curr_trial+1) > threshold)
			if n_blinks >= n_flicks:
				list_to_rem.append(i)
		return list_to_rem

	def displacement_vec(self, x, t, s=25, w=25, t_0=685):
		#take the difference of the 50 ms before saccade onset and 50-100 ms after saccade onset
		r_all = []
		to_rem_list = []
		for i in range(len(t)):
			#print(t[i]-w, t[i])
			if t[i] < t_0:
				to_rem_list.append(i)
			
			x_old = np.nanmean(x[i, t[i]-w:t[i], :], axis=0)
			x_new = np.nanmean(x[i, t[i]+s:t[i]+s+w, :], axis=0)
			r = x_new - x_old
			r_all.append(np.expand_dims(r, axis=0))
		
		r_all = np.concatenate(r_all)
		return r_all, to_rem_list

#single element
def convert_to_polar_1d(eye):
	# shape is (x, y)
	r = np.sqrt(np.sum(eye**2))
	theta = np.arctan2(eye[1], eye[0])
	return np.array([r, theta])

def plot_eye_data(eye_data):
	eye_proc_obj = prepEyeTracking()
	
	eye_proc_obj.plot_time_series(eye_data[:, :, 0], "t (in ms)", "position (in pix)", "x-position of the eye", "./Results/EEGNet_LSTM_WM/stats/gf_eye_x_all.png")
	eye_proc_obj.plot_time_series(eye_data[:, :, 1], "t (in ms)", "position (in pix)", "y-position of the eye", "./Results/EEGNet_LSTM_WM/stats/gf_eye_y_all.png")
	
	rand_sel = np.random.choice(eye_data.shape[0], int(0.01*eye_data.shape[0]), replace=False)
	
	eye_proc_obj.plot_time_series(eye_data[rand_sel, :, 0], "t (in ms)", "position (in pix)", "x-position of the eye", "./Results/EEGNet_LSTM_WM/stats/gf_eye_x_01p.png")
	eye_proc_obj.plot_time_series(eye_data[rand_sel, :, 1], "t (in ms)", "position (in pix)", "y-position of the eye", "./Results/EEGNet_LSTM_WM/stats/gf_eye_y_01p.png")

	rand_sel = np.random.choice(eye_data.shape[0], 10, replace=False)
	eye_proc_obj.plot_time_series(eye_data[rand_sel, :, 0], "t (in ms)", "position (in pix)", "x-position of the eye", "./Results/EEGNet_LSTM_WM/stats/gf_eye_x_10.png")
	eye_proc_obj.plot_time_series(eye_data[rand_sel, :, 1], "t (in ms)", "position (in pix)", "y-position of the eye", "./Results/EEGNet_LSTM_WM/stats/gf_eye_y_10.png")

def load_dispEye_saccade():
	eye_preproc = prepEyeTracking()
	eeg_data, eye_data = read_eeg_gaze_data()
	eye_data = eye_preproc.gaussian_smoothing_eye(np.expand_dims(eye_data, axis=0))
	plt.plot(eye_data[0, :, 0], label='x')
	plt.plot(eye_data[0, :, 1], label='y')
	plt.legend()
	plt.savefig("./data_eeg_gaze_data/processing/plots/trail.png")
	plt.close()

	print(eeg_data.shape, eye_data.shape)
	return

	eye_proc_obj = prepEyeTracking()
	pos_eye_data = eye_proc_obj.conv_to_pix(read_eye_tracking_WM("./data_deepak/eye-tracking/"))
	pos_eye_data = eye_proc_obj.gaussian_smoothing_eye(pos_eye_data)
	
	acc_eye_data = eye_proc_obj.magnitude_of_acc_from_pos(pos_eye_data, 0.002)
	speed_eye_data = eye_proc_obj.magnitude_of_vel_from_pos(pos_eye_data, 0.002)
	eye_proc_obj.plot_velocity_hist(speed_eye_data, True, 1000, "log(velocity) (in pix/s)", "Density", "Velocity", "./Results/EEGNet_LSTM_WM/stats/vel_distr.png")
	eye_proc_obj.plot_velocity_hist(acc_eye_data, True, 1000, "log(accelaration) (in pix/s^2)", "Density", "Accelaration", "./Results/EEGNet_LSTM_WM/stats/acc_distr.png")

	list_to_rem = eye_proc_obj.rem_blinks(acc_eye_data, 14.5, 6)   # numbers chosen from the distribution
	print(len(list_to_rem))
	pos_eye_data = np.delete(pos_eye_data, list_to_rem, axis=0)
	print(pos_eye_data.shape)
	#time dt -> 2 ms, c
	#plot_eye_data(pos_eye_data)

	saccade_onsets = eye_proc_obj.get_saccade_onset(speed_eye_data, 685, -15, 150)
	saccade_onsets = np.delete(saccade_onsets, list_to_rem, axis=0)

	eye_proc_obj.plot_velocity_hist(saccade_onsets*2, False, 100, "t (in ms)", "Density", "Saccade onset distribution", "./Results/EEGNet_LSTM_WM/stats/saccade_onsets.png")

	disp_eye, _ = eye_proc_obj.displacement_vec(pos_eye_data, saccade_onsets, s=38, w=50)

	#print(disp_eye.shape)

	eye_proc_obj.plot_velocity_hist(disp_eye[:, 0], False, 100, "Displacement x (in pix)", "Density", "Del x", "./Results/EEGNet_LSTM_WM/stats/del_x.png")
	eye_proc_obj.plot_velocity_hist(disp_eye[:, 1], False, 100, "Displacement y (in pix)", "Density", "Del y", "./Results/EEGNet_LSTM_WM/stats/del_y.png")
	# #return disp_eye

# load_dispEye_saccade()

def get_saccade_data():
	eye_preproc = prepEyeTracking()
	eeg_dat, eye_dat, block_id = read_data_withEvent()
	print(len(eeg_dat), len(eye_dat), block_id.shape)
	sacc_details = []
	bid_details = []
	eeg_details = []
	for i in range(len(eeg_dat)):
		event_marking = eye_dat[i][:, 2]
		eeg_dat_file = eeg_dat[i]
		#eye_dat_file = eye_preproc.gaussian_smoothing_eye(np.expand_dims(eye_dat[i][:, 0:2], axis=0))[0]
		eye_dat_file = eye_dat[i][:, 0:2]
		# plt.plot(eye_dat_file[:, 0], label='x')
		# plt.plot(eye_dat_file[:, 1], label='y')
		# plt.legend()
		# plt.savefig("./data_eeg_gaze_data/processing/plots/trail.png")
		bid = block_id[i]
		# print(event_marking.shape, eeg_dat_file.shape, eye_dat_file.shape, bid)
		# print(eeg_dat_file.shape, eye_dat_file.shape)
		
		# saccade_locs = np.where(event_marking==2)[0]
		
		vel_file = np.log(eye_preproc.magnitude_of_vel_from_pos(np.expand_dims(eye_dat_file, axis=0), 0.02)[0] + 1)
		acc_file = np.log(eye_preproc.magnitude_of_acc_from_pos(np.expand_dims(eye_dat_file, axis=0), 0.02)[0] + 1)
		
		saccade_locs = np.where((vel_file > 8.) * (acc_file > 12) * (acc_file < 13.5))[0]
		diff_saccade_loc = np.roll(saccade_locs, -1) - saccade_locs
		
		#remove adjacent saccade cluster - select the first
		# print(saccade_locs)
		saccade_locs = np.delete(saccade_locs, np.where(diff_saccade_loc < 7)[0])
		diff_saccade_loc = np.roll(saccade_locs, -1) - saccade_locs
		
		scale_factor = eeg_dat_file.shape[1] / eye_dat_file.shape[0]
		# print(scale_factor, eeg_dat_file.shape[1] - eye_dat_file.shape[0]*10)
		#ensure that only one event occurs at a time
		for i in range(1, len(saccade_locs) - 1):
			s_t = saccade_locs[i]
			avail_dat_left = diff_saccade_loc[i-1]
			avail_dat_right = diff_saccade_loc[i]
			#if avail_dat_left + avail_dat_right >= 49:

				#check eeg if it has all non-nan values in this range
			if (avail_dat_left >= 25) and (avail_dat_right >= 25):
				eeg_start_i, eeg_end_i = int(s_t*scale_factor - 250), 500 + int(s_t*scale_factor - 250)
			
				# print("Saccade time: {}, no_event_left = {}, no_event right: {}, eeg index left: {}, eeg index right: {}".format(s_t, avail_dat_left, avail_dat_right, eeg_start_i, eeg_end_i))
			# 	elif (avail_dat_right >= 25):
			# 		 eeg_start_i, eeg_end_i = int((s_t - avail_dat_left)*scale_factor), 500 + int((s_t - avail_dat_left)*scale_factor)
			# 	else:
			# 		eeg_start_i, eeg_end_i = int((s_t + avail_dat_right)*scale_factor) - 500, int((s_t + avail_dat_right)*scale_factor)
				req_eeg_i = eeg_dat_file[:, eeg_start_i:eeg_end_i]
				if np.sum(np.isnan(req_eeg_i)) == 0 and req_eeg_i.shape[1] == 500: #and (10*s_t > eeg_start_i and 10*s_t < eeg_end_i):
					from_pos_i, to_pos_i = s_t - 25, s_t + 20
					prev_eye_pos_i = eye_dat_file[from_pos_i : from_pos_i + 5]
					final_eye_pos_i = eye_dat_file[to_pos_i : to_pos_i + 5]
					if np.sum(np.isnan(prev_eye_pos_i)) + np.sum(np.isnan(final_eye_pos_i)) == 0:
						eye_pos_from_i, eye_pos_to_i = np.mean(prev_eye_pos_i, axis=0), np.mean(final_eye_pos_i, axis=0)
						# print(eye_pos_from_i.shape)
						sacc_disp_v = convert_to_polar_1d(eye_pos_to_i - eye_pos_from_i)
						
						if sacc_disp_v[0] > 50: 		#only in case of a valid saccade
							eeg_details.append([req_eeg_i])
							sacc_details.append([sacc_disp_v])
							bid_details.append(bid)
		#print(eye_pos_from_i.shape, eye_pos_to_i.shape)
		# print(np.unique(diff_saccade_loc))
		# print(saccade_locs, diff_saccade_loc)
	
	sacc_details = np.concatenate(sacc_details)
	bid_details = np.array(bid_details)
	eeg_details = np.concatenate(eeg_details)
	print(sacc_details.shape, bid_details.shape, eeg_details.shape)
	np.savez('./data_eeg_gaze_data/processing/saccade_data/saccade_prep_data.npz', eeg_data=eeg_details.astype('float32'), eye_data=sacc_details.astype('float32'), block_id=bid_details.astype('int32'))
	print("Saved Data")

# get_saccade_data()

def leave_sub_out(block_id, kfold):
    np.random.seed(99)
    train_list, test_list = [], []
    parts = np.unique(block_id) #--- 22
    n_parts = len(parts)
    # print(len(np.unique(block_id))) -- 22
    # selected ordering - 4, 4, 5, 5, test - 4
    test_list_ids = np.random.choice(np.arange(n_parts), kfold, replace=False)
    train_list_ids = np.delete(np.arange(n_parts), test_list_ids)
    
    #print(test_list_ids, train_list_ids, parts)
    
    test_list_ids = list(parts[test_list_ids])
    train_list_ids = list(parts[train_list_ids])

    train_list = [tno for (tno, index) in enumerate(block_id) if index in train_list_ids]
    test_list = [tno for (tno, index) in enumerate(block_id) if index in test_list_ids]

    #print(np.unique(train_list), np.unique(test_list))
    #print(len(train_list), len(test_list), len(list(set(train_list) & set(test_list))))
    return train_list, test_list

def get_eye_only_npz():
	full_file = np.load('./data_eeg_gaze_data/processing/saccade_data/saccade_prep_data.npz')
	eye_data = full_file['eye_data']
	req_loc = np.where(eye_data[:, 0] > 50)[0]
	return eye_data[req_loc]

def read_data_npz():
	obj_eye = prepEyeTracking()
	full_file = np.load('./data_eeg_gaze_data/processing/saccade_data/fix_prep_data.npz')
	eeg_data = np.expand_dims(full_file['eeg_data'], axis=-1)
	eye_data = full_file['eye_data']
	block_id = full_file['block_id']

	# req_loc = np.where((eye_data[:, 0] > 50) * (eye_data[:, 0] < 300))[0]
	# eeg_data = eeg_data[req_loc]
	# eye_data = eye_data[req_loc]
	# block_id = block_id[req_loc]

	train_list, test_list = leave_sub_out(block_id, kfold=5)
	
	# print(eeg_data.shape, eye_data.shape, block_id.shape)
	# obj_eye.plot_velocity_hist(eye_data[req_loc,0], logscale=False, nbins=1000, xlabel_="Saccade magnitude", y_label="Density", title_="Eye-Gaze dataset", saveloc='./data_eeg_gaze_data/processing/plots/r_g50.png')
	# obj_eye.plot_velocity_hist(eye_data[req_loc,1], logscale=False, nbins=1000, xlabel_="Theta magnitude", y_label="Density", title_="Eye-Gaze dataset", saveloc='./data_eeg_gaze_data/processing/plots/theta_g50.png')
	print(eye_data[train_list].shape, eye_data[test_list].shape)
	return eeg_data, eye_data, train_list, test_list
# read_data_npz()

def find_saccade():
	eye_preproc = prepEyeTracking()
	eeg_dat, eye_dat, block_id = read_data_withEvent()
	sacc_details = []
	bid_details = []
	eeg_details = []

	vel_distr = []
	acc_distr = []
	for i in range(len(eeg_dat)):
		event_marking = eye_dat[i][:, 2]
		eeg_dat_file = eeg_dat[i]
		eye_dat_file = eye_dat[i][:, 0:2]
		bid = block_id[i]
		
		vel_file = eye_preproc.magnitude_of_vel_from_pos(np.expand_dims(eye_dat_file, axis=0), 0.02)[0]
		acc_file = eye_preproc.magnitude_of_acc_from_pos(np.expand_dims(eye_dat_file, axis=0), 0.02)[0]

		vel_distr += list(vel_file)
		acc_distr += list(acc_file)
		# print(event_marking.shape, eeg_dat_file.shape, eye_dat_file.shape, bid.shape, vel_file.shape, acc_file.shape)
		
	vel_distr = np.array(vel_distr)
	acc_distr = np.array(acc_distr)
	print(vel_distr.shape, acc_distr.shape, flush=True)
	eye_preproc.plot_velocity_hist(vel_distr, logscale=True, nbins=100, xlabel_="Velocity in pix/s (log scale)", y_label="Density", title_="Velocity distribution", saveloc='./Results/eeg_robots_conservative/stats/eye_vel.png')
	eye_preproc.plot_velocity_hist(acc_distr, logscale=True, nbins=100, xlabel_="Acc. in pix/s^2 (log scale)", y_label="Density", title_="Accelaration distribution", saveloc='./Results/eeg_robots_conservative/stats/eye_acc.png')
	# sacc_details = np.concatenate(sacc_details)
	# bid_details = np.array(bid_details)
	# eeg_details = np.concatenate(eeg_details)
	# print(sacc_details.shape, bid_details.shape, eeg_details.shape)
	# np.savez('./data_eeg_gaze_data/processing/saccade_data/saccade_prep_data.npz', eeg_data=eeg_details.astype('float32'), eye_data=sacc_details.astype('float32'), block_id=bid_details.astype('int32'))
	# print("Saved Data")

# find_saccade()

def plot_saccade_locations():
	eye_preproc = prepEyeTracking()
	_, eye_dat, _ = read_data_withEvent()
	
	event_marking = eye_dat[0][:, 2]
	eye_dat_file = eye_dat[0][:, 0:2]
	
	vel_file = np.log(eye_preproc.magnitude_of_vel_from_pos(np.expand_dims(eye_dat_file, axis=0), 0.02)[0] + 1)
	acc_file = np.log(eye_preproc.magnitude_of_acc_from_pos(np.expand_dims(eye_dat_file, axis=0), 0.02)[0] + 1)
	
	saccade_locs = np.where((vel_file > 8.) * (acc_file > 12) * (acc_file < 13.5))[0]
	diff_saccade_loc = np.roll(saccade_locs, -1) - saccade_locs
	
	saccade_locs = np.delete(saccade_locs, np.where(diff_saccade_loc < 7)[0])
	diff_saccade_loc = np.roll(saccade_locs, -1) - saccade_locs

	# time = 1000 + np.arange(500)
	# plt.plot(time, eye_dat_file[1000:1500, 0], label='x')
	# plt.plot(time, eye_dat_file[1000:1500, 1], label='y')
	plt.plot(np.sqrt(np.sum(eye_dat_file**2, axis=1)), label='r')
	
	#plt.plot(eye_dat_file[:, 1], label='y')

	for i in range(1, len(saccade_locs) - 1):
			s_t = saccade_locs[i]
			avail_dat_left = diff_saccade_loc[i-1]
			avail_dat_right = diff_saccade_loc[i]
			#if avail_dat_left + avail_dat_right >= 49:

			# check eeg if it has all non-nan values in this range
			if (avail_dat_left >= 25) and (avail_dat_right >= 25):  #and (s_t<1500 and s_t>1000): 
				from_pos_i, to_pos_i = s_t - 2, s_t + 2
				prev_eye_pos_i = eye_dat_file[from_pos_i-4 : from_pos_i]
				final_eye_pos_i = eye_dat_file[to_pos_i : to_pos_i+4]
				if np.sum(np.isnan(prev_eye_pos_i)) + np.sum(np.isnan(final_eye_pos_i)) == 0:
					eye_pos_from_i, eye_pos_to_i = np.mean(prev_eye_pos_i, axis=0), np.mean(final_eye_pos_i, axis=0)
					# print(eye_pos_from_i.shape)
					sacc_disp_v = convert_to_polar_1d(eye_pos_to_i - eye_pos_from_i)
					if sacc_disp_v[0] > 50: 		#only in case of a valid saccade
						plt.axvline(x=s_t, color='k', linewidth=0.5, linestyle='--')

	plt.legend()
	plt.savefig("./Results/eeg_robots_conservative_velacc/stats/trail_sacc_plot.png")
	plt.close()


 
# plot_saccade_locations()
def get_fixations_data():
	eye_preproc = prepEyeTracking()
	eeg_dat, eye_dat, block_id = read_data_withEvent()
	print(len(eeg_dat), len(eye_dat), block_id.shape)
	fix_details = []
	bid_details = []
	eeg_details = []

	for i in range(len(eeg_dat)):
		event_marking = eye_dat[i][:, 2]
		eeg_dat_file = eeg_dat[i]
		eye_dat_file = eye_dat[i][:, 0:2]
		bid = block_id[i]
		
		fixations = np.where(event_marking==1)[0]
		scale_factor = eeg_dat_file.shape[1] / eye_dat_file.shape[0]
		curr_trial_start = 0
		for j in range(len(fixations)):
			s_t = fixations[j]
			# curr_trial_start tell me the minimum start of an allowed fixation 
			if s_t < curr_trial_start:
				continue
			elif (s_t > eye_dat_file.shape[0] - 100):	#not enough fixation to consider
				break
			#check eeg if it has all non-nan values in this range - 2 sec fixation will give us 1 sec data
			if np.sum(event_marking[s_t : s_t + 100] == 1) == 100:
				eeg_start_i, eeg_end_i = int((s_t+25)*scale_factor), 500 + int((s_t+25)*scale_factor)
				
				req_eeg_i = eeg_dat_file[:, eeg_start_i:eeg_end_i]
				if np.sum(np.isnan(req_eeg_i)) == 0 and req_eeg_i.shape[1] == 500: #and (10*s_t > eeg_start_i and 10*s_t < eeg_end_i):
					from_pos_i, to_pos_i = s_t + 25, s_t + 75
					eye_position_i = np.expand_dims(np.nanmean(eye_dat_file[from_pos_i : to_pos_i], axis=0), axis=0)
					
					eeg_details.append([req_eeg_i])
					fix_details.append(eye_position_i)
					bid_details.append(bid)

					curr_trial_start = s_t + 100
		
	fix_details = np.concatenate(fix_details, axis=0)
	bid_details = np.array(bid_details)
	eeg_details = np.concatenate(eeg_details)
	print(fix_details.shape, bid_details.shape, eeg_details.shape)
	np.savez('./data_eeg_gaze_data/processing/saccade_data/fix_prep_data.npz', eeg_data=eeg_details.astype('float32'), eye_data=fix_details.astype('float32'), block_id=bid_details.astype('int32'))
	print("Saved Data")

# get_fixations_data()