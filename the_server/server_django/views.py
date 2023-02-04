from django.http import HttpResponse
import os
import json
from .settings import BASE_DIR
from django.views.decorators.csrf import csrf_exempt
from the_server.utils.mp_to_bvh_solution import BvhSolution
from the_server.utils.video2mp_np import video2mp_np
from the_server.utils.whole_solution import WholeSolution
import the_server.server_django as server_django


@csrf_exempt
def video2bvh(request):
    # deprecated, do not use it!
    if request.method == 'POST':

        # save the video to server
        file2store = request.FILES['file']
        dir_videos = None
        if file2store:
            dir_videos = os.path.join(os.path.join(BASE_DIR, 'temp'), 'videos')

            destination = open(os.path.join(dir_videos, file2store.name), 'wb+')
            for chunk in file2store.chunks():
                destination.write(chunk)
            # destination.write(file2store)
            destination.close()

        # video to bvh and writes it into a file
        config_dir = os.path.join(BASE_DIR, 'configs')
        bvh = BvhSolution(
            os.path.join(config_dir, 'bvh_mp_config_final.json'),
            os.path.join(config_dir, 'mp_hierarchy.json'),
            os.path.join(config_dir, 'my5.bvh')
        )
        tmp1 = video2mp_np(os.path.join(dir_videos, file2store.name))
        mp_data, frame_rate = tmp1['mp_data'], tmp1['frame_rate']

        bvh.convert_mediapipe(mp_data)
        bvh.write_bvh(frame_rate, os.path.join(BASE_DIR, 'static', 'bvh_files', 'output.bvh'))

        with open(os.path.join(BASE_DIR, 'static', 'bvh_files', 'output.bvh'), 'r') as f:
            bvh_text = f.read()

        consumption = float(request.GET['weight']) * float(request.GET['span']) / 60 / 60 * 1.05 * 8

        # generate the return json object
        return_dict = {
            'evaluations': {
                "scores": {
                    "holistic": 10,
                    "torso": 9,
                    "upper": 8,
                    "lower": 7,
                },
                "energy": {
                    "energy": consumption * 0.7,
                    "fat": consumption * 0.3,
                    "energy_standard": 110,
                    "fat_standard": 210
                }
            },
            'bvh': bvh_text,
        }

        # response_str=ReturnJson.get_json_str()
        # return HttpResponse(response_str)

        return HttpResponse(json.dumps(return_dict))


@csrf_exempt
def whole_service(request):
    # the entire service works here

    config_dir = os.path.join(BASE_DIR, 'configs')

    sol_tmp = WholeSolution(
        os.path.join(BASE_DIR, 'temp'),
        os.path.join(config_dir, 'bvh_mp_config_final.json'),
        os.path.join(config_dir, 'mp_hierarchy.json'),
        os.path.join(config_dir, 'my5.bvh'),
        os.path.join(config_dir, 'scoring_parts_bvh.json'),
        os.path.join(BASE_DIR, 'temp'),
        os.path.join(BASE_DIR, 'static/model_videos'),
        int(request.GET['type']),
        float(request.GET['span']),
        float(request.GET['weight'])
    )

    [ret, err_message, returned_json] = sol_tmp.robust_workflow(request)
    # todo: if you're so un-busy, writer a  logger to log it

    return HttpResponse(returned_json)
