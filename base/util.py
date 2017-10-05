import os
import copy
import pandas as pd
from dns.name import root


def clear_files(path):
	filelist=[]
	filelist=os.listdir(path)
	for f in filelist:
		filepath = os.path.join(path, f)
		if os.path.isfile(filepath):
			os.remove(filepath)
			print filepath+" removed!"
			
def keep_n_best_models(path, N = 3):
	filelist=[]
	filelist=os.listdir(path)
	
	best = []
	for f in filelist:
		filepath = os.path.join(path, f)
		
		
		if os.path.isfile(filepath):
			file = os.path.basename(filepath)
			val = int(file.split('.')[-2])
			if len(best) < N:
				best.append((val, filepath))
				best.sort(cmp=(lambda x,y:cmp(x[0],y[0])), reverse=True)
			elif val > best[-1][0]:
				best.append((val, filepath))
				best.sort(cmp=lambda x,y:cmp(x[0],y[0]), reverse=True)
				refile = best.pop()
				os.remove(refile[1])
				print refile[1] + " removed!"
			else:
				os.remove(filepath)
				print filepath + " removed!"
				
				
def submission(y_true, y_pred, flag_y, df_test, file = 'data/submission.csv'):
	sub = pd.DataFrame()
	sub['id'] = df_test['id']
	sub['is_affair_truth'] = y_true
	sub['is_affair'] = flag_y
	sub['affair_probability'] = y_pred
	print "Submission samples:%d, file:%s"%(len(df_test), file)
	sub.to_csv(file, index=False)

def gen_floders(root_name, floders=['src', 'data', 'pic', 'doc', 'model', 'logs', 'checkpoints']):
	
	if os.path.exists(root_name) :
		print 'Project %s already existed!'%(root_name)
	else:
		os.makedirs(root_name)
		map(lambda x: os.makedirs(root_name + "/" + x) , floders)
		print 'Project %s has been created!'%(root_name)

def make_text_of_main_import():
	imp = "import pandas as pd\n"
	imp = imp + "import data_process\n"
	imp = imp + "import numpy as np\n"
	imp = imp + "\n\n\n\n\n"
	return imp

def make_text_of_submission():
	fun = 'def submission(flag_y, df_test, file = \'../data/submission.csv\'):\n'
	fun = fun + '	sub = pd.DataFrame()\n'
	fun = fun + '	sub[\'id\'] = df_test[\'id\']\n'
	fun = fun + '	print "please implement!"\n'
	fun = fun + '	print "Submission samples:%d, file:%s"%(len(df_test), file)\n'
	fun = fun + '	sub.to_csv(file, index=False)\n'
	fun = fun + "\n\n\n"
	return fun

def make_text_of_experiment():
	fun = 'def experiment():\n'
	fun = fun + '	df_train = pd.read_csv("../data/train.csv")\n'
	fun = fun + '	df_test = pd.read_csv("../data/test.csv")\n'
	fun = fun + '	x_train, y_train, x_test, _ = data_process.gen_data(df_train, df_test)\n'
	fun = fun + '	print "please implement!"\n'
	fun = fun + "\n\n\n"
	return fun

def gen_main_file(root_name, content=[make_text_of_main_import(),'','','','',make_text_of_submission(),'','','',make_text_of_experiment(),'','','','','','','','','', \
					'if __name__ == \"__main__\":', '	experiment()','']):
	
	file = root_name + "/" + 'src/main.py'
	if os.path.exists(file) : 
		print 'File %s already existed!'%(file)
	else:
		f = open(file,'w')
		
		f.write(make_text_of_main_import())
		f.write(make_text_of_submission())
		f.write(make_text_of_experiment())
		write_lines(5, f)
		f.write('if __name__ == \"__main__\":\n')
		f.write('	experiment()\n')
		write_lines(3, f)
		f.close()
		print 'main file src/main.py has been created!'

def make_text_of_call_function():
	fun = 'def call_feature_func(data, feats_name, is_normalized=False):\n'
	fun = fun + '	feats = pd.DataFrame()\n'
	fun = fun + '	for name in feats_name:\n'
	fun = fun + '		func_name = "extract_{feat}_feature".format(feat = name)\n'
	fun = fun + '		feats[name] = eval(func_name)(data, is_normalized)\n'
	fun = fun + '	return feats\n'
	return fun

def make_text_of_extract_function(col):
	fun = 'def extract_{col}_feature(data, is_normalized=False):\n'.format(col = col)
	
	fun = fun + '	def q_val(val):\n		return val\n'
	fun = fun + '	vals = data.apply(lambda x: q_val(x[\'{col}\']), axis=1, raw=True)\n'.format(col = col)
	fun = fun + '	if is_normalized:\n'
	fun = fun + '		vals = standardization(vals)\n'
	fun = fun + '	return vals\n'
	fun = fun + '\n\n\n'
	return fun

def make_text_of_extract_obj_function():
	fun = 'def extract_y_info(data):\n'
	fun = fun + '	print \'Need to be implemented\'\n'
	fun = fun + '	return None\n'
	return fun

def make_text_of_extract_gen_function():
	fun = 'def gen_data(df_train, df_test, is_resample=False, is_normalized=False):\n'
	fun = fun + '	x_train = extract_all_features([df_train], is_normalized)\n'
	fun = fun + '	y_train = extract_y_info(df_train)\n'
	fun = fun + '\n\n\n'
	fun = fun + '	x_test = extract_all_features([df_test], is_normalized)\n'
	fun = fun + '	y_test = extract_y_info(df_test)\n'
	fun = fun + '\n\n\n'
	fun = fun + '	x_t = x_train\n'
	fun = fun + '	y_t = y_train\n'
	fun = fun + '	if is_resample:\n'
	fun = fun + '		x_t, y_t = data_util.resample(x_train, y_train)\n'
	fun = fun + '	return x_t, y_t, x_test, y_test\n'
	fun = fun + '\n\n\n'
	return fun
	
def make_text_of_extract_all_function(cols, flg_col=None):
	fun = 'def extract_all_features(df_list, is_normalized=False):\n'
	fun = fun + '	data_all = pd.concat(df_list)\n'
	fun = fun + '	feats_name = [\n'
	for c in cols:
		if flg_col is None or c != flg_col:
			fun = fun + '				\'{col}\',\n'.format(col = c)
	fun = fun + '				]\n'
	fun = fun + '\n'
	fun = fun + '	feats = call_feature_func(data_all, feats_name, is_normalized)\n'
	fun = fun + '	return feats\n'
	fun = fun + '\n\n\n'
	return fun


	
def write_lines(n, f):
	for i in range(n):
		f.write('\n')

def gen_data_process_file(root_name, columns, flg_col=None, train_file='train.csv'):
	
	file = root_name + "/" + 'src/data_process.py'
	if os.path.exists(file) : 
		print 'File %s already existed!'%(file)
	else:
		f = open(file,'w')
		f.write('import pandas as pd\n')
		f.write('import numpy as np\n')
		f.write('import sys\n')
		f.write('import matplotlib.pyplot as plt\n')
		f.write('from sklearn import preprocessing\n')
		f.write('from sklearn.cross_validation import train_test_split\n')
		f.write('from sklearn import metrics\n')
		f.write('from collections import Counter\n')
		f.write('import seaborn as sns\n')
		f.write('from sklearn import preprocessing\n')
		f.write('sys.path.append(\'../../\')\n')
		f.write('from base import data_util\n')
		write_lines(5, f)
		
		f.write('def standardization(X):\n')
		f.write('	x_t = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)\n')
		f.write('	return x_t\n')
		
		f.write(make_text_of_call_function())
		
		write_lines(5, f)
		
		

		funcs = map(lambda x: make_text_of_extract_function(x), columns) 
		map(lambda x: f.write(x), funcs)
		write_lines(5, f)
		
		f.write(make_text_of_extract_all_function(columns, flg_col))
		write_lines(5, f)
		f.write(make_text_of_extract_obj_function())
		write_lines(5, f)
		f.write(make_text_of_extract_gen_function())
		write_lines(5, f)
		f.write('def test():\n')
		f.write('	df_train = pd.read_csv(\'../data/{file}\')\n'.format(file=train_file))
		write_lines(5, f)
		f.write('if __name__ == \"__main__\":\n')
		f.write('	test()\n')
		f.close()
		print 'data_process file src/data_process.py has been created!'
	



def make_project(parent_name, name, cols):

	root_name = parent_name + '/' + name
	
	gen_floders(root_name)
	gen_main_file(root_name)
	gen_data_process_file(root_name, cols)
		

if __name__ == "__main__":
  	make_project("../", 'weekx', ['sentence_id','token_id','class','before','after'])
# 	gen_data_process_file('../week4', ['date_time','site_name','posa_continent','user_location_country','user_location_region','user_location_city','orig_destination_distance','user_id','is_mobile','is_package','channel','srch_ci','srch_co','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id','srch_destination_type_id','hotel_continent','hotel_country','hotel_market','is_booking','cnt','hotel_cluster'])
#  	gen_data_process_file('../weekx', ['sentence_id','token_id','class','before','after'])