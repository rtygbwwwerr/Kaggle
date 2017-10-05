

class Config(object):
	batch_size = 85
	max_num_features = 31
	space_letter = 0
	boundary_letter = -1
	pad_size = 1
	
	dic_month = {"Jan":'january','Feb': 'february','Mar': 'march','Apr': 'april','May': 'may',
			'Jun': 'june','Jul': 'july','Jy': 'july','Aug': 'august','Sep': 'september',
			'Sept': 'september','Oct': 'october','Nov': 'november','Dec' : 'december',
			"1":'january','2': 'february','3': 'march','4': 'april','5': 'may',
			'6': 'june','7': 'july','8': 'august','9': 'september',
			'10': 'october','11': 'november','12' : 'december'}
	dic_week_day = {"Mon":'monday','Monday': 'monday','Tues': 'tuesday','Tuesday': 'tuesday','Wed': 'wednesday',
		'Wednesday': 'wednesday','Thur': 'thursday','Thursday': 'thursday','Fri': 'friday','Friday': 'friday',
		'Saturday': 'saturday','Sat': 'saturday','Sun': 'sunday','Sunday' : 'sunday'}

	special_dic_address = {'I00':'i zero', 
						   'B767':'b seven six seven', 
						   '3':'three',
						   '4':'four',
						    '5':'five', 
						    '6':'six',
						   '7':'seven',
						    '8':'eight', 
						    '9':'nigh',
						   '0':'o'}