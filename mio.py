# import codecs
# def load_string_list(file_path, is_utf8=False):
#     """
#     Load string list from mitok file
#     """
#     try:
#         if is_utf8:
#             f = codecs.open(file_path, 'r', 'utf-8')
#         else:
#             f = open(file_path)
#         l = []
#         for item in f:
#             item = item.strip()
#             if len(item) == 0:
#                 continue
#             l.append(item)
#         f.close()
#     except IOError:
#         print('open error %s' % file_path)
#         return None
#     else:
#         return l

import pandas as pd
def load_string_list(file_path):
    """
    Load string list from mitok file
    """
    l = pd.read_csv(file_path)['mask_img'].tolist()
    return l