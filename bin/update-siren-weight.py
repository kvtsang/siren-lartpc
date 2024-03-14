#!/usr/bin/env python3
import torch, sys, yaml, os

if not len(sys.argv) == 3:
    print(f'Usage: {sys.argv[0]}  check_point_file  config_file')
    sys.exit(1)

OUTPUT='updated.ckpt'
if os.path.isfile(OUTPUT):
    print('The output',OUTPUT,'already exists. Exiting.')
    sys.exit(2)
with open(sys.argv[1], 'rb') as fin:

    model_dict = torch.load(fin,map_location='cpu')

    print('Keys in the input file:')
    for k in model_dict.keys():
        print('  ',k)
    print()

    config=yaml.safe_load(open(sys.argv[2],'r').read())

    print('Keys in the config file:')
    for k in config.keys():
        print('  ',k)
    print()

    for key_to_pop in ['input_scale','output_scale','do_hardsigmoid','xform_cfg']:
        if key_to_pop in model_dict.keys():
            model_dict.pop(key_to_pop)

    model_dict['xform_cfg'] = config.get('transform_vis')
    model_dict['model_cfg'] = config['model']
    if 'scale' in model_dict['state_dict']:
        model_dict['state_dict']['output_scale'] = model_dict['state_dict'].pop('scale')

    print(f'Keys in the output ({OUTPUT}):')
    for k in model_dict:
        print(k)
    torch.save(model_dict,OUTPUT)
sys.exit(0)
