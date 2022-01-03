#ImageNet Image Classification Script of NXP BlueBox 2.0 using RTMaps.
import rtmaps.types
import numpy as np
from rtmaps.base_component import BaseComponent # base class
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os
import shutil
import time
import math
import warnings
import models
import matplotlib.pyplot as plt
from utils import convert_model, measure_model
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import time
from datetime import datetime

# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor

# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")

# Core() is called every time you have a new input
    def Core(self):

        parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
        parser.add_argument('data', metavar='DIR',
                            help='path to dataset')
        parser.add_argument('--model', default='condensenet', type=str, metavar='M',
                            help='model to train the dataset')
        parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=300, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=64, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                            metavar='LR', help='initial learning rate (default: 0.1)')
        parser.add_argument('--lr-type', default='cosine', type=str, metavar='T',
                            help='learning rate strategy (default: cosine)',
                            choices=['cosine', 'multistep'])
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--print-freq', '-p', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model (default: false)')
        parser.add_argument('--no-save-model', dest='no_save_model', action='store_true',
                            help='only save best model (default: false)')
        parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                            help='manual seed (default: 0)')
        parser.add_argument('--gpu', default=0,
                            help='gpu available')
        parser.add_argument('--savedir', type=str, metavar='PATH', default='results/savedir',
                            help='path to save result and checkpoint (default: results/savedir)')
        parser.add_argument('--resume', action='store_true',
                            help='use latest checkpoint if have any (default: none)')
        parser.add_argument('--stages', default=4-4-4, type=str, metavar='STAGE DEPTH',
                            help='per layer depth')
        parser.add_argument('--bottleneck', default=4, type=int, metavar='B',
                            help='bottleneck (default: 4)')
        parser.add_argument('--group-1x1', type=int, metavar='G', default=4,
                            help='1x1 group convolution (default: 4)')
        parser.add_argument('--group-3x3', type=int, metavar='G', default=4,
                            help='3x3 group convolution (default: 4)')
        parser.add_argument('--condense-factor', type=int, metavar='C', default=4,
                            help='condense factor (default: 4)')
        parser.add_argument('--growth', default=8-8-8, type=str, metavar='GROWTH RATE',
                            help='per layer growth')
        parser.add_argument('--reduction', default=0.5, type=float, metavar='R',
                            help='transition reduction (default: 0.5)')
        parser.add_argument('--dropout-rate', default=0, type=float,
                            help='drop out (default: 0)')
        parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO',
                            help='group lasso loss weight (default: 0)')
        parser.add_argument('--evaluate', action='store_true',
                            help='evaluate model on validation set (default: false)')
        parser.add_argument('--convert-from', default=None, type=str, metavar='PATH',
                            help='path to saved checkpoint (default: none)')
        parser.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                            help='path to saved checkpoint (default: none)')

        args = parser.parse_args(["--model", "condensenet", "-b", "256", "-j", "20", "imagenet", "--stages", "4-6-8-10-8", "--growth", "8-16-32-64-128", "--gpu", "0"])

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.stages = list(map(int, args.stages.split('-')))
        args.growth = list(map(int, args.growth.split('-')))
        if args.condense_factor is None:
            args.condense_factor = args.group_1x1

        if args.data == 'imagenet':
            args.num_classes = 1000
        elif args.data == 'cifar100':
            args.num_classes = 100
        else:
            args.num_classes = 10

        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                 std=[0.2675, 0.2565, 0.2761])
        train_set = transforms.Compose([
                                         transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         normalize,
                                             ])

        model = models.condensenet(args)
        model = nn.DataParallel(model)
        PATH = "results/path_to_the_trained_weights.pth.tar"

        model.load_state_dict(torch.load(PATH, map_location=torch.device("cpu"))['state_dict'])

        device = torch.device("cpu")
        model.eval()

        image = Image.open("test_image.jpg")
        #print(image.filename)
        print(f'Input file image: {image.filename}')

        input = train_set(image)
        input = input.unsqueeze(0)

        model.eval()
        output = model(input)        
        classes = ['abacus', 'abaya', 'academic_gown', 'accordion', 'acorn', 'acorn_squash', 'acoustic_guitar', 'admiral', 'affenpinscher', 'Afghan_hound', 'African_chameleon', 'African_crocodile', 'African_elephant', 'African_grey', 'African_hunting_dog', 'agama', 'agaric', 'aircraft_carrier', 'Airedale', 'airliner', 'airship', 'albatross', 'alligator_lizard', 'alp', 'altar', 'ambulance', 'American_alligator', 'American_black_bear', 'American_chameleon', 'American_coot', 'American_egret', 'American_lobster', 'American_Staffordshire_terrier', 'amphibian', 'analog_clock', 'anemone_fish', 'Angora', 'ant', 'apiary', 'Appenzeller', 'apron', 'Arabian_camel', 'Arctic_fox', 'armadillo', 'artichoke', 'ashcan', 'assault_rifle', 'Australian_terrier', 'axolotl', 'baboon', 'backpack', 'badger', 'bagel', 'bakery', 'balance_beam', 'bald_eagle', 'balloon', 'ballplayer', 'ballpoint', 'banana', 'Band_Aid', 'banded_gecko', 'banjo', 'bannister', 'barbell', 'barber_chair', 'barbershop', 'barn', 'barn_spider', 'barometer', 'barracouta', 'barrel', 'barrow', 'baseball', 'basenji', 'basketball', 'basset', 'bassinet', 'bassoon', 'bath_towel', 'bathing_cap', 'bathtub', 'beach_wagon', 'beacon', 'beagle', 'beaker', 'bearskin', 'beaver', 'Bedlington_terrier', 'bee', 'bee_eater', 'beer_bottle', 'beer_glass', 'bell_cote', 'bell_pepper', 'Bernese_mountain_dog', 'bib', 'bicycle-built-for-two', 'bighorn', 'bikini', 'binder', 'binoculars', 'birdhouse', 'bison', 'bittern', 'black_and_gold_garden_spider', 'black_grouse', 'black_stork', 'black_swan', 'black_widow', 'black-and-tan_coonhound', 'black-footed_ferret', 'Blenheim_spaniel', 'bloodhound', 'bluetick', 'boa_constrictor', 'boathouse', 'bobsled', 'bolete', 'bolo_tie', 'bonnet', 'book_jacket', 'bookcase', 'bookshop', 'Border_collie', 'Border_terrier', 'borzoi', 'Boston_bull', 'bottlecap', 'Bouvier_des_Flandres', 'bow', 'bow_tie', 'box_turtle', 'boxer', 'Brabancon_griffon', 'brain_coral', 'brambling', 'brass', 'brassiere', 'breakwater', 'breastplate', 'briard', 'Brittany_spaniel', 'broccoli', 'broom', 'brown_bear', 'bubble', 'bucket', 'buckeye', 'buckle', 'bulbul', 'bull_mastiff', 'bullet_train', 'bulletproof_vest', 'bullfrog', 'burrito', 'bustard', 'butcher_shop', 'butternut_squash', 'cab', 'cabbage_butterfly', 'cairn', 'caldron', 'can_opener', 'candle', 'cannon', 'canoe', 'capuchin', 'car_mirror', 'car_wheel', 'carbonara', 'Cardigan', 'cardigan', 'cardoon', 'carousel', 'carpenters_kit', 'carton', 'cash_machine', 'cassette', 'cassette_player', 'castle', 'catamaran', 'cauliflower', 'CD_player', 'cello', 'cellular_telephone', 'centipede', 'chain', 'chain_mail', 'chain_saw', 'chainlink_fence', 'chambered_nautilus', 'cheeseburger', 'cheetah', 'Chesapeake_Bay_retriever', 'chest', 'chickadee', 'chiffonier', 'Chihuahua', 'chime', 'chimpanzee', 'china_cabinet', 'chiton', 'chocolate_sauce', 'chow', 'Christmas_stocking', 'church', 'cicada', 'cinema', 'cleaver', 'cliff', 'cliff_dwelling', 'cloak', 'clog', 'clumber', 'cock', 'cocker_spaniel', 'cockroach', 'cocktail_shaker', 'coffee_mug', 'coffeepot', 'coho', 'coil', 'collie', 'colobus', 'combination_lock', 'comic_book', 'common_iguana', 'common_newt', 'computer_keyboard', 'conch', 'confectionery', 'consomme', 'container_ship', 'convertible', 'coral_fungus', 'coral_reef', 'corkscrew', 'corn', 'cornet', 'coucal', 'cougar', 'cowboy_boot', 'cowboy_hat', 'coyote', 'cradle', 'crane', 'crane', 'crash_helmet', 'crate', 'crayfish', 'crib', 'cricket', 'Crock_Pot', 'croquet_ball', 'crossword_puzzle', 'crutch', 'cucumber', 'cuirass', 'cup', 'curly-coated_retriever', 'custard_apple', 'daisy', 'dalmatian', 'dam', 'damselfly', 'Dandie_Dinmont', 'desk', 'desktop_computer', 'dhole', 'dial_telephone', 'diamondback', 'diaper', 'digital_clock', 'digital_watch', 'dingo', 'dining_table', 'dishrag', 'dishwasher', 'disk_brake', 'Doberman', 'dock', 'dogsled', 'dome', 'doormat', 'dough', 'dowitcher', 'dragonfly', 'drake', 'drilling_platform', 'drum', 'drumstick', 'dugong', 'dumbbell', 'dung_beetle', 'Dungeness_crab', 'Dutch_oven', 'ear', 'earthstar', 'echidna', 'eel', 'eft', 'eggnog', 'Egyptian_cat', 'electric_fan', 'electric_guitar', 'electric_locomotive', 'electric_ray', 'English_foxhound', 'English_setter', 'English_springer', 'entertainment_center', 'EntleBucher', 'envelope', 'Eskimo_dog', 'espresso', 'espresso_maker', 'European_fire_salamander', 'European_gallinule', 'face_powder', 'feather_boa', 'fiddler_crab', 'fig', 'file', 'fire_engine', 'fire_screen', 'fireboat', 'flagpole', 'flamingo', 'flat-coated_retriever', 'flatworm', 'flute', 'fly', 'folding_chair', 'football_helmet', 'forklift', 'fountain', 'fountain_pen', 'four-poster', 'fox_squirrel', 'freight_car', 'French_bulldog', 'French_horn', 'French_loaf', 'frilled_lizard', 'frying_pan', 'fur_coat', 'gar', 'garbage_truck', 'garden_spider', 'garter_snake', 'gas_pump', 'gasmask', 'gazelle', 'German_shepherd', 'German_short-haired_pointer', 'geyser', 'giant_panda', 'giant_schnauzer', 'gibbon', 'Gila_monster', 'goblet', 'go-kart', 'golden_retriever', 'goldfinch', 'goldfish', 'golf_ball', 'golfcart', 'gondola', 'gong', 'goose', 'Gordon_setter', 'gorilla', 'gown', 'grand_piano', 'Granny_Smith', 'grasshopper', 'Great_Dane', 'great_grey_owl', 'Great_Pyrenees', 'great_white_shark', 'Greater_Swiss_Mountain_dog', 'green_lizard', 'green_mamba', 'green_snake', 'greenhouse', 'grey_fox', 'grey_whale', 'grille', 'grocery_store', 'groenendael', 'groom', 'ground_beetle', 'guacamole', 'guenon', 'guillotine', 'guinea_pig', 'gyromitra', 'hair_slide', 'hair_spray', 'half_track', 'hammer', 'hammerhead', 'hamper', 'hamster', 'hand_blower', 'hand-held_computer', 'handkerchief', 'hard_disc', 'hare', 'harmonica', 'harp', 'hartebeest', 'harvester', 'harvestman', 'hatchet', 'hay', 'head_cabbage', 'hen', 'hen-of-the-woods', 'hermit_crab', 'hip', 'hippopotamus', 'hog', 'hognose_snake', 'holster', 'home_theater', 'honeycomb', 'hook', 'hoopskirt', 'horizontal_bar', 'hornbill', 'horned_viper', 'horse_cart', 'hot_pot', 'hotdog', 'hourglass', 'house_finch', 'howler_monkey', 'hummingbird', 'hyena', 'ibex', 'Ibizan_hound', 'ice_bear', 'ice_cream', 'ice_lolly', 'impala', 'Indian_cobra', 'Indian_elephant', 'indigo_bunting', 'indri', 'iPod', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'iron', 'isopod', 'Italian_greyhound', 'jacamar', 'jackfruit', 'jack-o-lantern', 'jaguar', 'Japanese_spaniel', 'jay', 'jean', 'jeep', 'jellyfish', 'jersey', 'jigsaw_puzzle', 'jinrikisha', 'joystick', 'junco', 'keeshond', 'kelpie', 'Kerry_blue_terrier', 'killer_whale', 'kimono', 'king_crab', 'king_penguin', 'king_snake', 'kit_fox', 'kite', 'knee_pad', 'knot', 'koala', 'Komodo_dragon', 'komondor', 'kuvasz', 'lab_coat', 'Labrador_retriever', 'lacewing', 'ladle', 'ladybug', 'Lakeland_terrier', 'lakeside', 'lampshade', 'langur', 'laptop', 'lawn_mower', 'leaf_beetle', 'leafhopper', 'leatherback_turtle', 'lemon', 'lens_cap', 'Leonberg', 'leopard', 'lesser_panda', 'letter_opener', 'Lhasa', 'library', 'lifeboat', 'lighter', 'limousine', 'limpkin', 'liner', 'lion', 'lionfish', 'lipstick', 'little_blue_heron', 'llama', 'Loafer', 'loggerhead', 'long-horned_beetle', 'lorikeet', 'lotion', 'loudspeaker', 'loupe', 'lumbermill', 'lycaenid', 'lynx', 'macaque', 'macaw', 'Madagascar_cat', 'magnetic_compass', 'magpie', 'mailbag', 'mailbox', 'maillot', 'maillot', 'malamute', 'malinois', 'Maltese_dog', 'manhole_cover', 'mantis', 'maraca', 'marimba', 'marmoset', 'marmot', 'mashed_potato', 'mask', 'matchstick', 'maypole', 'maze', 'measuring_cup', 'meat_loaf', 'medicine_chest', 'meerkat', 'megalith', 'menu', 'Mexican_hairless', 'microphone', 'microwave', 'military_uniform', 'milk_can', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'minibus', 'miniskirt', 'minivan', 'mink', 'missile', 'mitten', 'mixing_bowl', 'mobile_home', 'Model_T', 'modem', 'monarch', 'monastery', 'mongoose', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito_net', 'motor_scooter', 'mountain_bike', 'mountain_tent', 'mouse', 'mousetrap', 'moving_van', 'mud_turtle', 'mushroom', 'muzzle', 'nail', 'neck_brace', 'necklace', 'nematode', 'Newfoundland', 'night_snake', 'nipple', 'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier', 'notebook', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil_filter', 'Old_English_sheepdog', 'orange', 'orangutan', 'organ', 'oscilloscope', 'ostrich', 'otter', 'otterhound', 'overskirt', 'ox', 'oxcart', 'oxygen_mask', 'oystercatcher', 'packet', 'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama', 'palace', 'panpipe', 'paper_towel', 'papillon', 'parachute', 'parallel_bars', 'park_bench', 'parking_meter', 'partridge', 'passenger_car', 'patas', 'patio', 'pay-phone', 'peacock', 'pedestal', 'Pekinese', 'pelican', 'Pembroke', 'pencil_box', 'pencil_sharpener', 'perfume', 'Persian_cat', 'Petri_dish', 'photocopier', 'pick', 'pickelhaube', 'picket_fence', 'pickup', 'pier', 'piggy_bank', 'pill_bottle', 'pillow', 'pineapple', 'ping-pong_ball', 'pinwheel', 'pirate', 'pitcher', 'pizza', 'plane', 'planetarium', 'plastic_bag', 'plate', 'plate_rack', 'platypus', 'plow', 'plunger', 'Polaroid_camera', 'pole', 'polecat', 'police_van', 'pomegranate', 'Pomeranian', 'poncho', 'pool_table', 'pop_bottle', 'porcupine', 'pot', 'potpie', 'potters_wheel', 'power_drill', 'prairie_chicken', 'prayer_rug', 'pretzel', 'printer', 'prison', 'proboscis_monkey', 'projectile', 'projector', 'promontory', 'ptarmigan', 'puck', 'puffer', 'pug', 'punching_bag', 'purse', 'quail', 'quill', 'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio_telescope', 'rain_barrel', 'ram', 'rapeseed', 'recreational_vehicle', 'red_fox', 'red_wine', 'red_wolf', 'red-backed_sandpiper', 'redbone', 'red-breasted_merganser', 'redshank', 'reel', 'reflex_camera', 'refrigerator', 'remote_control', 'restaurant', 'revolver', 'rhinoceros_beetle', 'Rhodesian_ridgeback', 'rifle', 'ringlet', 'ringneck_snake', 'robin', 'rock_beauty', 'rock_crab', 'rock_python', 'rocking_chair', 'rotisserie', 'Rottweiler', 'rubber_eraser', 'ruddy_turnstone', 'ruffed_grouse', 'rugby_ball', 'rule', 'running_shoe', 'safe', 'safety_pin', 'Saint_Bernard', 'saltshaker', 'Saluki', 'Samoyed', 'sandal', 'sandbar', 'sarong', 'sax', 'scabbard', 'scale', 'schipperke', 'school_bus', 'schooner', 'scoreboard', 'scorpion', 'Scotch_terrier', 'Scottish_deerhound', 'screen', 'screw', 'screwdriver', 'scuba_diver', 'sea_anemone', 'sea_cucumber', 'sea_lion', 'sea_slug', 'sea_snake', 'sea_urchin', 'Sealyham_terrier', 'seashore', 'seat_belt', 'sewing_machine', 'Shetland_sheepdog', 'shield', 'Shih-Tzu', 'shoe_shop', 'shoji', 'shopping_basket', 'shopping_cart', 'shovel', 'shower_cap', 'shower_curtain', 'siamang', 'Siamese_cat', 'Siberian_husky', 'sidewinder', 'silky_terrier', 'ski', 'ski_mask', 'skunk', 'sleeping_bag', 'slide_rule', 'sliding_door', 'slot', 'sloth_bear', 'slug', 'snail', 'snorkel', 'snow_leopard', 'snowmobile', 'snowplow', 'soap_dispenser', 'soccer_ball', 'sock', 'soft-coated_wheaten_terrier', 'solar_dish', 'sombrero', 'sorrel', 'soup_bowl', 'space_bar', 'space_heater', 'space_shuttle', 'spaghetti_squash', 'spatula', 'speedboat', 'spider_monkey', 'spider_web', 'spindle', 'spiny_lobster', 'spoonbill', 'sports_car', 'spotlight', 'spotted_salamander', 'squirrel_monkey', 'Staffordshire_bullterrier', 'stage', 'standard_poodle', 'standard_schnauzer', 'starfish', 'steam_locomotive', 'steel_arch_bridge', 'steel_drum', 'stethoscope', 'stingray', 'stinkhorn', 'stole', 'stone_wall', 'stopwatch', 'stove', 'strainer', 'strawberry', 'street_sign', 'streetcar', 'stretcher', 'studio_couch', 'stupa', 'sturgeon', 'submarine', 'suit', 'sulphur_butterfly', 'sulphur-crested_cockatoo', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension_bridge', 'Sussex_spaniel', 'swab', 'sweatshirt', 'swimming_trunks', 'swing', 'switch', 'syringe', 'tabby', 'table_lamp', 'tailed_frog', 'tank', 'tape_player', 'tarantula', 'teapot', 'teddy', 'television', 'tench', 'tennis_ball', 'terrapin', 'thatch', 'theater_curtain', 'thimble', 'three-toed_sloth', 'thresher', 'throne', 'thunder_snake', 'Tibetan_mastiff', 'Tibetan_terrier', 'tick', 'tiger', 'tiger_beetle', 'tiger_cat', 'tiger_shark', 'tile_roof', 'timber_wolf', 'titi', 'toaster', 'tobacco_shop', 'toilet_seat', 'toilet_tissue', 'torch', 'totem_pole', 'toucan', 'tow_truck', 'toy_poodle', 'toy_terrier', 'toyshop', 'tractor', 'traffic_light', 'trailer_truck', 'tray', 'tree_frog', 'trench_coat', 'triceratops', 'tricycle', 'trifle', 'trilobite', 'trimaran', 'tripod', 'triumphal_arch', 'trolleybus', 'trombone', 'tub', 'turnstile', 'tusker', 'typewriter_keyboard', 'umbrella', 'unicycle', 'upright', 'vacuum', 'valley', 'vase', 'vault', 'velvet', 'vending_machine', 'vestment', 'viaduct', 'vine_snake', 'violin', 'vizsla', 'volcano', 'volleyball', 'vulture', 'waffle_iron', 'Walker_hound', 'walking_stick', 'wall_clock', 'wallaby', 'wallet', 'wardrobe', 'warplane', 'warthog', 'washbasin', 'washer', 'water_bottle', 'water_buffalo', 'water_jug', 'water_ouzel', 'water_snake', 'water_tower', 'weasel', 'web_site', 'weevil', 'Weimaraner', 'Welsh_springer_spaniel', 'West_Highland_white_terrier', 'whippet', 'whiptail', 'whiskey_jug', 'whistle', 'white_stork', 'white_wolf', 'wig', 'wild_boar', 'window_screen', 'window_shade', 'Windsor_tie', 'wine_bottle', 'wing', 'wire-haired_fox_terrier', 'wok', 'wolf_spider', 'wombat', 'wood_rabbit', 'wooden_spoon', 'wool', 'worm_fence', 'wreck', 'yawl', 'yellow_ladys_slipper', 'Yorkshire_terrier', 'yurt', 'zebra', 'zucchini']
        topk=(1,5)
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = pred[0].cpu().numpy()[0]
        pred = classes[pred]
        #print(pred)
        print(f'Prediction: {pred}')

# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass