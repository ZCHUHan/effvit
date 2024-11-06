import os
import random
import numpy as np

from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter


    
class ADE20KSegmentation(data.Dataset):
    ADEClass = namedtuple('ADEClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                       'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        ADEClass(name='wall', id=1, train_id=1, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[120, 120, 120]) ,
        ADEClass(name='building, edifice', id=2, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[180, 120, 120]) ,
        ADEClass(name='sky', id=3, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[6, 230, 230]) ,
        ADEClass(name='floor, flooring', id=4, train_id=4, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[80, 50, 50]) ,
        ADEClass(name='tree', id=5, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[4, 200, 3]) ,
        ADEClass(name='ceiling', id=6, train_id=6, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[120, 120, 80]) ,
        ADEClass(name='road, route', id=7, train_id=7, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[140, 140, 140]) ,
        ADEClass(name='bed', id=8, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[204, 5, 255]) ,
        ADEClass(name='windowpane, window', id=9, train_id=9, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[230, 230, 230]) ,
        ADEClass(name='grass', id=10, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[4, 250, 7]) ,
        ADEClass(name='cabinet', id=11, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[224, 5, 255]) ,
        ADEClass(name='sidewalk, pavement', id=12, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[235, 255, 7]) ,
        ADEClass(name='person, individual, someone, somebody, mortal, soul', id=13, train_id=13, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[150, 5, 61]) ,
        ADEClass(name='earth, ground', id=14, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[120, 120, 70]) ,
        ADEClass(name='door, double door', id=15, train_id=15, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[8, 255, 51]) ,
        ADEClass(name='table', id=16, train_id=16, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 6, 82]) ,
        ADEClass(name='mountain, mount', id=17, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[143, 255, 140]) ,
        ADEClass(name='plant, flora, plant life', id=18, train_id=18, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[204, 255, 4]) ,
        ADEClass(name='curtain, drape, drapery, mantle, pall', id=19, train_id=19, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 51, 7]) ,
        ADEClass(name='chair', id=20, train_id=20, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[204, 70, 3]) ,
        ADEClass(name='car, auto, automobile, machine, motorcar', id=21, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 102, 200]) ,
        ADEClass(name='water', id=22, train_id=22, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[61, 230, 250]) ,
        ADEClass(name='painting, picture', id=23, train_id=23, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 6, 51]) ,
        ADEClass(name='sofa, couch, lounge', id=24, train_id=24, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[11, 102, 255]) ,
        ADEClass(name='shelf', id=25, train_id=25, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 7, 71]) ,
        ADEClass(name='house', id=26, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 9, 224]) ,
        ADEClass(name='sea', id=27, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[9, 7, 230]) ,
        ADEClass(name='mirror', id=28, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[220, 220, 220]) ,
        ADEClass(name='rug, carpet, carpeting', id=29, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 9, 92]) ,
        ADEClass(name='field', id=30, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[112, 9, 255]) ,
        ADEClass(name='armchair', id=31, train_id=31, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[8, 255, 214]) ,
        ADEClass(name='seat', id=32, train_id=32, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[7, 255, 224]) ,
        ADEClass(name='fence, fencing', id=33, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 184, 6]) ,
        ADEClass(name='desk', id=34, train_id=34, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[10, 255, 71]) ,
        ADEClass(name='rock, stone', id=35, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 41, 10]) ,
        ADEClass(name='wardrobe, closet, press', id=36, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[7, 255, 255]) ,
        ADEClass(name='lamp', id=37, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[224, 255, 8]) ,
        ADEClass(name='bathtub, bathing tub, bath, tub', id=38, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[102, 8, 255]) ,
        ADEClass(name='railing, rail', id=39, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 61, 6]) ,
        ADEClass(name='cushion', id=40, train_id=40, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 194, 7]) ,
        ADEClass(name='base, pedestal, stand', id=41, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 122, 8]) ,
        ADEClass(name='box', id=42, train_id=42, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[0, 255, 20]) ,
        ADEClass(name='column, pillar', id=43, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 8, 41]) ,
        ADEClass(name='signboard, sign', id=44, train_id=44, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 5, 153]) ,
        ADEClass(name='chest of drawers, chest, bureau, dresser', id=45, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[6, 51, 255]) ,
        ADEClass(name='counter', id=46, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[235, 12, 255]) ,
        ADEClass(name='sand', id=47, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[160, 150, 20]) ,
        ADEClass(name='sink', id=48, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 163, 255]) ,
        ADEClass(name='skyscraper', id=49, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[140, 140, 140]) ,
        ADEClass(name='fireplace, hearth, open fireplace', id=50, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[250, 10, 15]) ,
        ADEClass(name='refrigerator, icebox', id=51, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[20, 255, 0]) ,
        ADEClass(name='grandstand, covered stand', id=52, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[31, 255, 0]) ,
        ADEClass(name='path', id=53, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 31, 0]) ,
        ADEClass(name='stairs, steps', id=54, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 224, 0]) ,
        ADEClass(name='runway', id=55, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[153, 255, 0]) ,
        ADEClass(name='case, display case, showcase, vitrine', id=56, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 0, 255]) ,
        ADEClass(name='pool table, billiard table, snooker table', id=57, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 71, 0]) ,
        ADEClass(name='pillow', id=58, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 235, 255]) ,
        ADEClass(name='screen door, screen', id=59, train_id=59, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[0, 173, 255]) ,
        ADEClass(name='stairway, staircase', id=60, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[31, 0, 255]) ,
        ADEClass(name='river', id=61, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[11, 200, 200]) ,
        ADEClass(name='bridge, span', id=62, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 82, 0]) ,
        ADEClass(name='bookcase', id=63, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 255, 245]) ,
        ADEClass(name='blind, screen', id=64, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 61, 255]) ,
        ADEClass(name='coffee table, cocktail table', id=65, train_id=65, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[0, 255, 112]) ,
        ADEClass(name='toilet, can, commode, crapper, pot, potty, stool, throne', id=66, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 255, 133]) ,
        ADEClass(name='flower', id=67, train_id=67, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 0, 0]) ,
        ADEClass(name='book', id=68, train_id=68, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 163, 0]) ,
        ADEClass(name='hill', id=69, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 102, 0]) ,
        ADEClass(name='bench', id=70, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[194, 255, 0]) ,
        ADEClass(name='countertop', id=71, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 143, 255]) ,
        ADEClass(name='stove, kitchen stove, range, kitchen range, cooking stove', id=72, train_id=72, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[51, 255, 0]) ,
        ADEClass(name='palm, palm tree', id=73, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 82, 255]) ,
        ADEClass(name='kitchen island', id=74, train_id=74, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[0, 255, 41]) ,
        ADEClass(name='computer, computing machine, computing device, data processor, electronic computer, information processing system', id=75, train_id=75, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[0, 255, 173]) ,
        ADEClass(name='swivel chair', id=76, train_id=76, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[10, 0, 255]) ,
        ADEClass(name='boat', id=77, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[173, 255, 0]) ,
        ADEClass(name='bar', id=78, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 255, 153]) ,
        ADEClass(name='arcade machine', id=79, train_id=79, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 92, 0]) ,
        ADEClass(name='hovel, hut, hutch, shack, shanty', id=80, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 0, 255]) ,
        ADEClass(name='bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle', id=81, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 0, 245]) ,
        ADEClass(name='towel', id=82, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 0, 102]) ,
        ADEClass(name='light, light source', id=83, train_id=83, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 173, 0]) ,
        ADEClass(name='truck, motortruck', id=84, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 0, 20]) ,
        ADEClass(name='tower', id=85, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 184, 184]) ,
        ADEClass(name='chandelier, pendant, pendent', id=86, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 31, 255]) ,
        ADEClass(name='awning, sunshade, sunblind', id=87, train_id=87, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[0, 255, 61]) ,
        ADEClass(name='streetlight, street lamp', id=88, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 71, 255]) ,
        ADEClass(name='booth, cubicle, stall, kiosk', id=89, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 0, 204]) ,
        ADEClass(name='television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box', id=90, train_id=90, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[0, 255, 194]) ,
        ADEClass(name='airplane, aeroplane, plane', id=91, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 255, 82]) ,
        ADEClass(name='dirt track', id=92, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 10, 255]) ,
        ADEClass(name='apparel, wearing apparel, dress, clothes', id=93, train_id=93, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[0, 112, 255]) ,
        ADEClass(name='pole', id=94, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[51, 0, 255]) ,
        ADEClass(name='land, ground, soil', id=95, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 194, 255]) ,
        ADEClass(name='bannister, banister, balustrade, balusters, handrail', id=96, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 122, 255]) ,
        ADEClass(name='escalator, moving staircase, moving stairway', id=97, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 255, 163]) ,
        ADEClass(name='ottoman, pouf, pouffe, puff, hassock', id=98, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 153, 0]) ,
        ADEClass(name='bottle', id=99, train_id=99, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[0, 255, 10]) ,
        ADEClass(name='buffet, counter, sideboard', id=100, train_id=100, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 112, 0]) ,
        ADEClass(name='poster, posting, placard, notice, bill, card', id=101, train_id=101, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[143, 255, 0]) ,
        ADEClass(name='stage', id=102, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[82, 0, 255]) ,
        ADEClass(name='van', id=103, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[163, 255, 0]) ,
        ADEClass(name='ship', id=104, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 235, 0]) ,
        ADEClass(name='fountain', id=105, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[8, 184, 170]) ,
        ADEClass(name='conveyer belt, conveyor belt, conveyer, conveyor, transporter', id=106, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[133, 0, 255]) ,
        ADEClass(name='canopy', id=107, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 255, 92]) ,
        ADEClass(name='washer, automatic washer, washing machine', id=108, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[184, 0, 255]) ,
        ADEClass(name='plaything, toy', id=109, train_id=109, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 0, 31]) ,
        ADEClass(name='swimming pool, swimming bath, natatorium', id=110, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 184, 255]) ,
        ADEClass(name='stool', id=111, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 214, 255]) ,
        ADEClass(name='barrel, cask', id=112, train_id=112, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 0, 112]) ,
        ADEClass(name='basket, handbasket', id=113, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[92, 255, 0]) ,
        ADEClass(name='waterfall, falls', id=114, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 224, 255]) ,
        ADEClass(name='tent, collapsible shelter', id=115, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[112, 224, 255]) ,
        ADEClass(name='bag', id=116, train_id=116, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[70, 184, 160]) ,
        ADEClass(name='minibike, motorbike', id=117, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[163, 0, 255]) ,
        ADEClass(name='cradle', id=118, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[153, 0, 255]) ,
        ADEClass(name='oven', id=119, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[71, 255, 0]) ,
        ADEClass(name='ball', id=120, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 0, 163]) ,
        ADEClass(name='food, solid food', id=121, train_id=121, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[255, 204, 0]) ,
        ADEClass(name='step, stair', id=122, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 0, 143]) ,
        ADEClass(name='tank, storage tank', id=123, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 255, 235]) ,
        ADEClass(name='trade name, brand name, brand, marque', id=124, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[133, 255, 0]) ,
        ADEClass(name='microwave, microwave oven', id=125, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 0, 235]) ,
        ADEClass(name='pot, flowerpot', id=126, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[245, 0, 255]) ,
        ADEClass(name='animal, animate being, beast, brute, creature, fauna', id=127, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 0, 122]) ,
        ADEClass(name='bicycle, bike, wheel, cycle', id=128, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 245, 0]) ,
        ADEClass(name='lake', id=129, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[10, 190, 212]) ,
        ADEClass(name='dishwasher, dish washer, dishwashing machine', id=130, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[214, 255, 0]) ,
        ADEClass(name='screen, silver screen, projection screen', id=131, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 204, 255]) ,
        ADEClass(name='blanket, cover', id=132, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[20, 0, 255]) ,
        ADEClass(name='sculpture', id=133, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 255, 0]) ,
        ADEClass(name='hood, exhaust hood', id=134, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 153, 255]) ,
        ADEClass(name='sconce', id=135, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 41, 255]) ,
        ADEClass(name='vase', id=136, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 255, 204]) ,
        ADEClass(name='traffic light, traffic signal, stoplight', id=137, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[41, 0, 255]) ,
        ADEClass(name='tray', id=138, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[41, 255, 0]) ,
        ADEClass(name='ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', id=139, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[173, 0, 255]) ,
        ADEClass(name='fan', id=140, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 245, 255]) ,
        ADEClass(name='pier, wharf, wharfage, dock', id=141, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[71, 0, 255]) ,
        ADEClass(name='crt screen', id=142, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[122, 0, 255]) ,
        ADEClass(name='plate', id=143, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 255, 184]) ,
        ADEClass(name='monitor, monitoring device', id=144, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 92, 255]) ,
        ADEClass(name='bulletin board, notice board', id=145, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[184, 255, 0]) ,
        ADEClass(name='shower', id=146, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[0, 133, 255]) ,
        ADEClass(name='radiator', id=147, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[255, 214, 0]) ,
        ADEClass(name='glass, drinking glass', id=148, train_id=148, category='object', category_id=1, has_instances=False, ignore_in_eval=False, color=[25, 194, 194]) ,
        ADEClass(name='clock', id=149, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[102, 255, 0]) ,
        ADEClass(name='flag', id=150, train_id=255, category='void', category_id=0, has_instances=False, ignore_in_eval=True, color=[92, 0, 255]) ,
    ]
    
    
    valid_classes = [1, 4, 6, 7, 9, 13, 15, 16, 18, 19, 20, 22, 23, 24, 25, 
                     31, 32, 34, 40, 42, 44, 59, 65, 67, 68, 72, 74, 75, 76,
                     79, 83, 87, 90, 93, 99, 100, 101, 109, 112, 116, 121, 148]
    
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    NUM_CLASS = 42 #150

    def __init__(self, root='../datasets/ade', split='test', mode=None, transform=None):
        self.root = os.path.expanduser(root)
        self.images, self.masks = _get_ade20k_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))
    
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = cls.NUM_CLASS
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ("wall", "building, edifice", "sky", "floor, flooring", "tree",
                "ceiling", "road, route", "bed", "windowpane, window", "grass",
                "cabinet", "sidewalk, pavement",
                "person, individual, someone, somebody, mortal, soul",
                "earth, ground", "door, double door", "table", "mountain, mount",
                "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",
                "chair", "car, auto, automobile, machine, motorcar",
                "water", "painting, picture", "sofa, couch, lounge", "shelf",
                "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",
                "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",
                "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",
                "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
                "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",
                "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",
                "grandstand, covered stand", "path", "stairs, steps", "runway",
                "case, display case, showcase, vitrine",
                "pool table, billiard table, snooker table", "pillow",
                "screen door, screen", "stairway, staircase", "river", "bridge, span",
                "bookcase", "blind, screen", "coffee table, cocktail table",
                "toilet, can, commode, crapper, pot, potty, stool, throne",
                "flower", "book", "hill", "bench", "countertop",
                "stove, kitchen stove, range, kitchen range, cooking stove",
                "palm, palm tree", "kitchen island",
                "computer, computing machine, computing device, data processor, "
                "electronic computer, information processing system",
                "swivel chair", "boat", "bar", "arcade machine",
                "hovel, hut, hutch, shack, shanty",
                "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
                "motorcoach, omnibus, passenger vehicle",
                "towel", "light, light source", "truck, motortruck", "tower",
                "chandelier, pendant, pendent", "awning, sunshade, sunblind",
                "streetlight, street lamp", "booth, cubicle, stall, kiosk",
                "television receiver, television, television set, tv, tv set, idiot "
                "box, boob tube, telly, goggle box",
                "airplane, aeroplane, plane", "dirt track",
                "apparel, wearing apparel, dress, clothes",
                "pole", "land, ground, soil",
                "bannister, banister, balustrade, balusters, handrail",
                "escalator, moving staircase, moving stairway",
                "ottoman, pouf, pouffe, puff, hassock",
                "bottle", "buffet, counter, sideboard",
                "poster, posting, placard, notice, bill, card",
                "stage", "van", "ship", "fountain",
                "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
                "canopy", "washer, automatic washer, washing machine",
                "plaything, toy", "swimming pool, swimming bath, natatorium",
                "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",
                "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",
                "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
                "trade name, brand name, brand, marque", "microwave, microwave oven",
                "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",
                "bicycle, bike, wheel, cycle", "lake",
                "dishwasher, dish washer, dishwashing machine",
                "screen, silver screen, projection screen",
                "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
                "traffic light, traffic signal, stoplight", "tray",
                "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
                "dustbin, trash barrel, trash bin",
                "fan", "pier, wharf, wharfage, dock", "crt screen",
                "plate", "monitor, monitoring device", "bulletin board, notice board",
                "shower", "radiator", "glass, drinking glass", "clock", "flag")


def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print('cannot find the mask:', maskpath)

    return img_paths, mask_paths