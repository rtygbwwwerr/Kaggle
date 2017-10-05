
import pandas as pd
import re
from matplotlib.testing.jpl_units import day
import inflect
from config import Config as cfg


def num_to_words(str, group=2):
	
	p = inflect.engine()
	words = p.number_to_words(str, group=group, andword='')
	words = words.replace(', ', ' ')
	words = words.replace('-', ' ')
	return words

def num_to_order(str):
	p = inflect.engine()
	str = num_to_words(str, group=0)
	words = p.ordinal(str)
	words = words.replace(', ', ' ')
	words = words.replace('-', ' ')
	return words
	
def extract_years(text):
	p_years = re.compile("([1-9][0-9]?[0-9]?[0-9]?)(\s?|'?)(s|A\.?D\.?|B\.?C\.?)")
	p_year = re.compile("([1-9][0-9][0-9][0-9])")
# 	p_year_end = re.compile(r'(-|/)([1-9][0-9][0-9]?[0-9]?)$')
	
	
	
	year = None
	year_info = None
	matched = p_years.match(text)
	if matched:
		year = matched.group(1)
		year_info = matched.group(3)
#         print matched.group(3)
	elif p_year.match(text):
		matched = p_year.match(text)
		year = matched.group(1)
	else :
		matched = re.findall(r'[-/][1-9][0-9][0-9]?[0-9]?$', text, re.M)
		if matched and len(matched) > 0:
			year = filter(str.isdigit, matched[0])
		

	return year, year_info
		
def extract_month(text):
	p_month = re.compile("(Jan|Feb|Mar|Apr|May|Jun|Jul|Jy|Aug|Sep|Sept|Oct|Nov|Dec)")
	p_month_head = re.compile("([0-9]?[0-9])(\.|/)([0-9]?[0-9])")
	p_month_mid = re.compile("([1-9][0-9][0-9][0-9])(-)([0-9]?[0-9])")
	matched = p_month.match(text)
	month = None
	if matched:
		month = matched.group(1)
	elif p_month_head.match(text):
		matched = p_month_head.match(text)
		month = matched.group(1)
	elif p_month_mid.match(text):
		matched = p_month_mid.match(text)
		month = matched.group(3)
	if month is not None and month.isdigit():
		month = str(int(month))
	return month	

def extract_day(text):
	p_head = re.compile("([0-9]?[0-9])( |-|th )(Jan|Feb|Mar|Apr|May|Jun|Jul|Jy|Aug|Sep|Sept|Oct|Nov|Dec)")
	p_mid = re.compile("([0-9]?[0-9])(/|\.)([0-9]?[0-9])")
	p_last_1 = re.compile("([1-9][0-9][0-9][0-9])-([0-9]?[0-9])-([0-9]?[0-9])")
	p_last_2 = re.compile("(January|February|March|April|May|June|July|August|September|October|November|December)(\.? )([0-9]?[0-9])")
	p_last_3 = re.compile("(Jan|Feb|Mar|Apr|May|Jun|Jul|Jy|Aug|Sep|Sept|Oct|Nov|Dec)(\.? )([0-9]?[0-9])")
	
	is_day_first = True
	matched = p_head.match(text)
	day = None
	if matched:
		day = matched.group(1)
	elif p_mid.match(text):
		matched = p_mid.match(text)
		day = matched.group(3)
	elif p_last_1.match(text):
		matched = p_last_1.match(text)
		day = matched.group(3)
	elif p_last_2.match(text):
		matched = p_last_2.match(text)
		day = matched.group(3)
		is_day_first = False
	elif p_last_3.match(text):
		matched = p_last_3.match(text)
		day = matched.group(3)
		is_day_first = False
	return day, is_day_first

def extract_week_day(text):
	p_head = re.compile("(Mon|Monday|Tues|Tuesday|Wed|Wednesday|Thur|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)")
	matched = p_head.match(text)
	day = None
	if matched:
		day = matched.group(1)
	return day

def check_and_correct(day, month):
	
	im = 0
	id = 0
	day_after = day
	month_after = month

	if month is not None and month.isdigit():
		im = int(month)
		month = str(im)
	if day is not None and day.isdigit():
		id = int(day)
		day = str(id)
		
	if im > 12 and im < 32:
		day_after = month
		month_after = day
	return day_after, month_after

def extract_date_info(text):
	year, year_info = extract_years(text)
	month = extract_month(text)
	day, is_day_first = extract_day(text)
	week_day = extract_week_day(text)
	
	day, month = check_and_correct(day, month)
	
# 	print "before:%s"%(text)
# 	print "year:%s, info:%s"%(year, year_info)
# 	print "month:%s"%(month)
# 	print "day:%s, is_first:%s"%(day, str(is_day_first))
# 	print "week_day:%s"%(week_day)
	info = {}
	info['year'] = year
	info['month'] = month
	info['day'] = day
	info['week_day'] = week_day
	info['year_info'] = year_info
	info['is_day_first'] = is_day_first
	
	return info
def pluralize(text):
	p = inflect.engine()
	after = p.plural_noun(text)
	
	return after
def convert_year(text, year_info):
	after = ""
	head = ""
	if text is not None:
		if text.startswith('200'):
			after = num_to_words(text, 0)
		else:
			after = num_to_words(text)

	if text is not None and year_info is not None:
		if year_info != "s":
			year_info = year_info.replace(".", "")
			year_info = year_info.replace(" ", "")
			year_info = ' '.join(list(year_info.lower()))
			year_info = ' ' + year_info
			
			if len(text) == 1:
				head = 'o '
	else:
		year_info = ""
		
	if year_info == "s":
		after =	pluralize(after)
		year_info = ""
		
	after = head + after + year_info
	return after
def convert_month(text):
	after = ""
	if text is not None:
		text = text.replace('.', "")
		after = cfg.dic_month[text]
	return after

def convert_day(text, is_day_first):
	after = ""
	if text is not None:
		after = num_to_order(text)
		if is_day_first:
			after = "the " + after + ' of'
	return after

def convert_week_day(text):
	after = ""
	if text is not None:
		text = text.replace('.', "")
		after = cfg.dic_week_day[text]
	return after

def convert(before):
	info = extract_date_info(before)
	year = convert_year(info['year'], info['year_info'])
	month = convert_month(info['month'])
	day = convert_day(info['day'], info['is_day_first'])
	week_day = convert_week_day(info['week_day'])
	
	if info['is_day_first']:
		after = '{arg_week_day} {arg_day} {arg_month} {arg_year}'.format(arg_week_day = week_day, arg_day = day, arg_month = month, arg_year = year)
	else :
		after = '{arg_week_day} {arg_month} {arg_day} {arg_year}'.format(arg_week_day = week_day, arg_day = day, arg_month = month, arg_year = year)
	after = after.strip()
	return after

def test():
	df = pd.read_csv("../data/grouped/DATE_order.csv")
	df['after_t'] = df['before'].apply(lambda x : convert(x))
	del df['len']
	df_err = df[df['after']!=df['after_t']]
	print "acc:%f"%((len(df) - len(df_err)) / float(len(df)))
	df.to_csv('../data/DATE_order_t.csv', index=False)
	df_err.to_csv('../data/DATE_order_err.csv', index=False)
if __name__ == "__main__":
	test()
#Dec. 15th	
#15th December
# 	after = convert('4AD.')
# 	print after