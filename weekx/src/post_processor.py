# -*- coding: UTF-8 -*-
import re
import sys

reload(sys)
sys.setdefaultencoding('utf8')


class RuleBasedNormalizer(object):

    URL_RE_STR = "([a-zA-Z\-]{2,}\.[a-zA-Z\-]{2,}|::|#\w+|\.[a-zA-Z\-]{2,}|^(\d{1,3}\.){3}\d{1,3}|www|[a-zA-Z]:\/\/|ELECTRONICS|ELECTRONIC|([\da-fA-F]{1,2}[\-:]){5}[\da-fA-F]{1,2})|([\da-fA-F]{1,4}[:]){7}([\da-fA-F]{1,4})"

    HASH_TAG_RE_STR = "^#\w+$"

    COPY_TOKENS = ['::']

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

    SPECIAL_CHAR_DICT = {
        "ˈ": ("single quote", True),
        "#": ("hash", True),
        "%": ("percen", True),
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

    def __init__(self):
        self.url_re_obj = re.compile(self.URL_RE_STR)
        self.hash_re_ojb = re.compile(self.HASH_TAG_RE_STR)

    def normalize(self, orig_s):
        if self.url_re_obj.search(orig_s):
            modify, norm_s = self._normalize_url(orig_s)
            return (modify, norm_s)
        else:
            return (False, '')

    def _normalize_url(self, orig_s):
        if orig_s in self.COPY_TOKENS:
            return (True, orig_s)
        elif self.hash_re_ojb.match(orig_s):
            return (False, '')
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
                    normed_list.extend(self._change_number(number_s))
                    i = j - 1
                else:
                    normed_list.append(c)
                i += 1

            return (True, ' '.join(normed_list))

    def _split_str(self, s):
        return [x for x in list(s) if x.isalpha()]

    def _change_number(self, s):
        assert s.isdigit()
        if len(s) == 2 and s in self.TWO_DIGIT_MAP:
            return self._split_str(self.TWO_DIGIT_MAP[s])
        else:
            l = []
            for digit in s:
                l.extend(self._split_str(self.ONE_DIGIT_MAP[digit]))
            return l

if __name__ == '__main__':
    norm = RuleBasedNormalizer()
    for line in sys.stdin:
        line = line.decode('utf8').strip()
        flag, res = norm.normalize(line)
        if flag:
            print '1\t%s\t%s' % (line, res)
        else:
            print '0\t%s\t%s' % (line, '')
