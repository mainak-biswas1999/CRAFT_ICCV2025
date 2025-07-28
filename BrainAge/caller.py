from TASFAR_sfcn import *
from craft_semisup import *

if __name__ == "__main__":
	# perc_labels = [0.4, 0.6]
	# st_randoms=[2, 0]
	# ctr = 0
	# for perc_label in perc_labels:
	# 	call_k_fold_tlsa("./Results/rebuttal/tasfar_perc/", "./Models/rebuttal/tasfar_perc/", perc_label=perc_label, pretrain_loc="./Models/rebuttal/perc_sup/"+str(int(100*perc_label))+"/", check_train=False, st_random=st_randoms[ctr])
	# 	ctr = ctr + 1		

	perc_labels = [0.6]
	for perc_label in perc_labels:
		# caller_paper_subsamp_kseed("./Results/rebuttal/craft_perc/", "./Models/rebuttal/craft_perc/", perc_label=perc_label, pretrain_loc="./Models/rebuttal/perc_sup/"+str(int(100*perc_label))+"/", check_train=False)
		caller_paper_subsamp_kseed("./Results/rebuttal/craft_perc_ukb/", "./Models/rebuttal/craft_perc_ukb/", perc_label=perc_label, pretrain_loc=None, check_train=False)