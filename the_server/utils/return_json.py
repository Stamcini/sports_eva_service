# deprecated!!!




# import scoring
#
#
# class ReturnJson:
#     default = -1
#     score_items = ['holistic', 'torso', 'upper', 'lower']
#     energy_items = ['energy', 'fat', 'energy_standard', 'fat_standard']
#     return_dict_default = {
#         'evaluations': {
#             "scores": {
#                 "holistic": default,
#                 "torso": default,
#                 "upper": default,
#                 "lower": default,
#             },
#             "energy": {
#                 "energy": default,
#                 "fat": default,
#                 "energy_standard": default,
#                 "fat_standard": default
#             }
#         },
#         'bvh': default,
#     }
#
#     @staticmethod
#     def get_json_str(ret: bool = False, ) -> str:
#         import json
#         import copy
#         returned_dict = copy.deepcopy(ReturnJson.return_dict_default)
#         if not ret:
#             pass
#         else:
#             for item in ReturnJson.score_items:
#                 returned_dict['evaluations']['scores'][item] = getattr(scoring.Scoring, item)()
#             for item in ReturnJson.energy_items:
#                 returned_dict['evaluations']['energy'][item] = getattr(scoring.Scoring, item)()
#
#         return json.dumps(returned_dict)
