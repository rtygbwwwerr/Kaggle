# -*- coding: UTF-8 -*-
import re
import sys

reload(sys)
sys.setdefaultencoding('utf8')


class RuleBasedNormalizer(object):

    URL_RE_STR = "([a-zA-Z\-]{2,}\.[a-zA-Z\-]{2,}|::|^#.+$|\.[a-zA-Z\-]{2,}|(\d{1,3}\.){3}\d{1,3}|www|[a-zA-Z]:\/\/|ELECTRONICS|ELECTRONIC|([\da-fA-F]{1,2}[\-:]){5}[\da-fA-F]{1,2})|([\da-fA-F]{1,4}[:]){7}([\da-fA-F]{1,4})"

    HASH_TAG_RE_STR = "^#[^\.]+$"

    COPY_TOKENS = ['::']

    MATH_AND_UNIT_MAP = {
        "⅝":("five eighths", "and five eighths"),
        "⅛":("one eighth", "and one eighth"),
        "½":("a half", "and a half"),
        "⅓":("one third", "and one third"),
        "¼":("a quarter", "and a quarter"),
        "⅞":("seven eighths", "and seven eighths"),
        "¾":("three quarters", "and three quarters"),
        "½in":("a half inches", "and a half inches"),
        "½ in":("a half inches", "and a half inches"),
        "½%":("a half percent", "and a half percent"),
        "½hp":("a half horsepower", "and a half horsepower"),
        "½ hp":("a half horsepower", "and a half horsepower"),
        "¾ m":("three quarters meters", "and three quarters meters"),
        "¾m":("three quarters meters", "and three quarters meters"),
        "kg/m²":"kilograms per square meter",
        "kg/m3":"kilograms per cubic meter",
        "kg/m³":"kilograms per cubic meter",
        "kj/m³":"kilo joules per cubic meter",
        "km²":"square kilometers",
        "/km²":"per square kilometers",
        "km³":"cubic kilometers",
        "km/h":"kilometers per hour",
        "µm":"micrometers",
        "m²":"square meters",
        "m³":"cubic meters",
        "m/s":"meters per second",
        "m³/s":"cubic meters per second",
        "mi²":"square miles",
        "/mi²":"per square miles",
        "mm²":" square millimeters",
        "μg":"micrograms",
        "μg/m3":"micrograms per cubic meters",
        "μg/m³":"micrograms per cubic meters",
        "μg/ml":"micrograms per milliliters",
        "μs":"microseconds",
        "kj/mol":"kilo joule per mole",
        "/cm²":"per square centimeter",
        "/cm2":"per square centimeter",
        "cm²":"square centimeter",
        "cm2":"square centimeter",
        "km/s":"kilometers per second",
    }

    ONE_DIGIT_MAP = {
        "0":"o",
        "1":"one",
        "2":"two",
        "3":"three",
        "4":"four",
        "5":"five",
        "6":"six",
        "7":"seven" ,
        "8":"eight",
        "9":"nine",
    }

    TWO_DIGIT_MAP = {
        "00":"o o",
        "01":"o one",
        "02":"o two",
        "03":"o three",
        "04":"o four",
        "05":"o five",
        "06":"o six",
        "07":"o seven",
        "08":"o eight",
        "09":"o nine",
        "10":"ten",
        "11":"eleven",
        "12":"twelve",
        "13":"thirteen",
        "14":"fourteen",
        "15":"fifteen",
        "16":"sixteen",
        "17":"seventeen",
        "18":"eighteen",
        "19":"nineteen",
        "20":"twenty",
        "21":"twenty one",
        "22":"twenty two",
        "23":"twenty three",
        "24":"twenty four",
        "25":"twenty five",
        "26":"twenty six",
        "27":"twenty seven",
        "28":"twenty eight",
        "29":"twenty nine",
        "30":"thirty",
        "31":"thirty one",
        "32":"thirty two",
        "33":"thirty three",
        "34":"thirty four",
        "35":"thirty five",
        "36":"thirty six",
        "37":"thirty seven",
        "38":"thirty eight",
        "39":"thirty nine",
        "40":"forty",
        "41":"forty one",
        "42":"forty two",
        "43":"forty three",
        "44":"forty four",
        "45":"forty five",
        "46":"forty six",
        "47":"forty seven",
        "48":"forty eight",
        "49":"forty nine",
        "50":"fifty",
        "51":"fifty one",
        "52":"fifty two",
        "53":"fifty three",
        "54":"fifty four",
        "55":"fifty five",
        "56":"fifty six",
        "57":"fifty seven",
        "58":"fifty eight",
        "59":"fifty nine",
        "60":"sixty",
        "61":"sixty one",
        "62":"sixty two",
        "63":"sixty three",
        "64":"sixty four",
        "65":"sixty five",
        "66":"sixty six",
        "67":"sixty seven",
        "68":"sixty eight",
        "69":"sixty nine",
        "70":"seventy",
        "71":"seventy one",
        "72":"seventy two",
        "73":"seventy three",
        "74":"seventy four",
        "75":"seventy five",
        "76":"seventy six",
        "77":"seventy seven",
        "78":"seventy eight",
        "79":"seventy nine",
        "80":"eighty",
        "81":"eighty one",
        "82":"eighty two",
        "83":"eighty three",
        "84":"eighty four",
        "85":"eighty five",
        "86":"eighty six",
        "87":"eighty seven",
        "88":"eighty eight",
        "89":"eighty nine",
        "90":"ninety",
        "91":"ninety one",
        "92":"ninety two",
        "93":"ninety three",
        "94":"ninety four",
        "95":"ninety five",
        "96":"ninety six",
        "97":"ninety seven",
        "98":"ninety eight",
        "99":"ninety nine",
    }

    THOUSAND_NUM_MAP = {
        "2000":"two thousand",
        "2001":"two thousand one",
        "2002":"two thousand two",
        "2003":"two thousand three",
        "2004":"two thousand four",
        "2005":"two thousand five",
        "2006":"two thousand six",
        "2007":"two thousand seven",
        "2008":"two thousand eight",
        "2009":"two thousand nine",
    }

    SPECIAL_CHAR_DICT = {
        "ˈ": ("single quote", True),
        "#": ("hash", True),
        "%": ("percent", True),
        ")": ("closing parenthesis", True),
        "(": ("opening parenthesis", True),
        "-": ("dash", False),
        "/": ("slash", False),
        ".": ("dot", False),
        ";": ("semicolon", True),
        ":": ("colon", False),
        "_": ("under score", True),
        "~": ("tilde", True),
    }

    def __init__(self, special_norm_file='../data/special_norm_map.list'):
        self.url_re_obj = re.compile(self.URL_RE_STR)
        self.hash_re_ojb = re.compile(self.HASH_TAG_RE_STR)
        self.special_norm_map = {}
        if special_norm_file:
            with open(special_norm_file) as fo:
                for line in fo:
                    parts = line.strip().decode('utf8').split('\t')
                    if len(parts) != 3:
                        continue
                    if parts[1] and parts[2]:
                        self.special_norm_map[parts[1]] = parts[2]
            print >>sys.stderr, "Loaded %d special norm rules from file." % len(self.special_norm_map)

        join_unit_re_str = '(' + '|'.join(self.MATH_AND_UNIT_MAP.keys()) + ')'
        self.split_unit_re_obj = re.compile(u'^(.*?)(%s)$' % join_unit_re_str)

    def pre_split(self, orig_s):
        orig_s = orig_s.decode('utf8').lower()
        m = self.split_unit_re_obj.match(orig_s)
        if not m:
            return (False, '', '')

        first_part = str(m.group(1)).strip()
        second_part = str(m.group(2)).strip()
        if second_part not in self.MATH_AND_UNIT_MAP:
            return (False, '', '')

        norm_rule = self.MATH_AND_UNIT_MAP[second_part]
        if len(norm_rule) == 1:
            suffix = norm_rule
        else:
            suffix = norm_rule[1] if first_part else norm_rule[0]
        return (True, first_part, suffix)

    def normalize(self, orig_s):
        orig_s = orig_s.decode('utf8')
        '''
        # rule2.
        if orig_s in self.special_norm_map:
            return (True, self.special_norm_map[orig_s])
        '''

        # rule3.
        if len(orig_s) == 1 and orig_s >= u'\u4e00' and orig_s <= u'\u9fa5':
            return (True, orig_s)

        # rule4 url.
        if self.url_re_obj.search(orig_s):
            modify, norm_s = self._normalize_url(orig_s)
            return (modify, norm_s)

        return (False, '')

    def _normalize_url(self, orig_s):
        if orig_s in self.COPY_TOKENS:
            return (True, orig_s)
        elif self.hash_re_ojb.match(orig_s):
            return self._normalize_hash_tag(orig_s)
        else:
            normed_list = []
            char_list = list(orig_s.lower())
            i = 0
            while i < len(char_list):
                c = char_list[i]
                if c in self.SPECIAL_CHAR_DICT:
                    map_s, _split = self.SPECIAL_CHAR_DICT[c]
                    l = [map_s]
                    if _split:
                        l = self._split_str(map_s)
                    normed_list.extend(l)
                elif c.isdigit():
                    j = i
                    number_s = ''
                    while j < len(char_list) and char_list[j].isdigit():
                        number_s += char_list[j]
                        j += 1
                    norm_num_s = self._norm_number(number_s)
                    normed_list.extend(self._split_str(norm_num_s))
                    i = j - 1
                else:
                    normed_list.append(c)
                i += 1

            return (True, ' '.join(normed_list))

    def _split_str(self, s):
        return [x for x in list(s) if x.isalpha()]

    def _norm_number(self, s):
        assert s.isdigit()
        num = int(s)
        if len(s) == 2 and s in self.TWO_DIGIT_MAP:
            return self.TWO_DIGIT_MAP[s]
        if (len(s) == 4 and
            ((num >= 1900 and num <= 1999) or
             (num >= 2010 and num <= 2017))):
            return self.TWO_DIGIT_MAP[s[0:2]] + ' ' + self.TWO_DIGIT_MAP[s[2:4]]
        if (len(s) == 4 and
            (num >= 2000 and num <= 2009) and
            s in self.THOUSAND_NUM_MAP):
            return self.THOUSAND_NUM_MAP[s]
        else:
            num_list = []
            for digit in s:
                num_list.append(self.ONE_DIGIT_MAP[digit])
            return ' '.join(num_list)

    def _normalize_hash_tag(self, s):
        content = s[1:]
        if content.isalpha():
            res = 'hash tag ' + content.lower()
            return (True, res)
        else:
            res_list = ['hash']
            i = 0
            while i < len(content):
                c = content[i]
                if c in self.SPECIAL_CHAR_DICT:
                    map_s, _ = self.SPECIAL_CHAR_DICT[c]
                    res_list.append(map_s)
                elif c.isdigit():
                    j = i
                    number_s = ''
                    while j < len(content) and content[j].isdigit():
                        number_s += content[j]
                        j += 1
                    norm_num_s = self._norm_number(number_s)
                    res_list.append(norm_num_s)
                    i = j - 1
                else:
                    j = i
                    temp_s = ''
                    while j < len(content) and content[j].isalpha():
                        temp_s += content[j]
                        j += 1
                    res_list.append(temp_s.lower())
                    i = j - 1
                i += 1
            return (True, ' '.join(res_list))

if __name__ == '__main__':
    norm = RuleBasedNormalizer()
    for line in sys.stdin:
        line = line.decode('utf8').strip()
        flag, res = norm.normalize(line)
        if flag:
            print '1\t%s\t%s' % (line, res)
        else:
            print '0\t%s\t%s' % (line, '')

