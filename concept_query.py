import time

import requests
import re
import json
import os
import pathlib

def init():
    sg = json.load(open('train_sceneGraphs.json'))
    sg_node_file = open('sg_node.txt', 'w')

    for image_idx in sg:
        # if image_idx[-4:]=='.png':
        #     image_idx = image_idx.replace('.png','')

        rel_name=[]
        obj_name=[]
        for obj in sg[image_idx]['objects']:
            if obj['name'] not in obj_name:
                obj_name.append(sg[image_idx]['objects'][obj]['name'])
                sg_node_file.write(obj['name']+'\n')
                for rel in sg[image_idx]['objects'][obj]['relations']:
                    if rel['name'] not in rel_name:
                        rel_name.append(rel['name'])
                        # print(rel['name'])
                        sg_node_file.write(rel['name'] + '\n')
        node_name=obj_name+rel_name
    sg_node_file.close()

def clean(input_str):
    string = input_str.lower()
    string = re.sub(r'a +(.*)', r'\1', string)
    string = re.sub(r'the +(.*)', r'\1', string)
    string = re.sub(r'an +(.*)', r'\1', string)
    return string

def img_object_node_name():
    sg = json.load(open('images_objects.json'))
    sg_concept = json.load(open('concept_sg.json'))

    for image_idx in sg:
        # if image_idx[-4:]=='.png':
        #     image_idx = image_idx.replace('.png','')
        concept_file = open(f'concepts/{image_idx}.txt', 'w')

        for node in sg[image_idx]:
            if node in sg_concept.keys():
                for triplet in sg_concept[node]:
                    # print(triplet)
                    concept_file.write(triplet+'\n')
        concept_file.close()
        if os.path.exists(f'concepts/{image_idx}.txt'):
            print(f'Processing {image_idx} ...')
            os.system(f'cp concepts/{image_idx}.txt data/ConceptNet/test.txt')
            os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 python src/run_kbc_subgraph.py --dataset conceptnet --sim_relations --bert_concat --use_bias --load_model model/conceptnet_best_subgraph_model.pth --eval_only --write_results >output_log.txt 2>&1')
            if os.path.exists('concept_embed.pkl'):
                pathlib.Path(f'concept_sg_embed/{image_idx}.pkl').parent.mkdir(parents=True, exist_ok=True)
                os.system(f'mv concept_embed.pkl concept_sg_embed/{image_idx}.pkl')
                if not os.path.exists(f'concept_sg_embed/{image_idx}.pkl'):
                    print('Failed to move the file.')
                else:
                    print('Done!')
            else:
                print('Embed file not found!')


def sg_node_name():
    sg = json.load(open('train_sceneGraphs.json'))
    sg_concept = json.load(open('concept_sg.json'))

    for image_idx in sg:
        # if image_idx[-4:]=='.png':
        #     image_idx = image_idx.replace('.png','')
        concept_file = open(f'{image_idx}.txt', 'w')

        rel_name=[]
        obj_name=[]
        for obj in sg[image_idx]['objects']:
            if sg[image_idx]['objects'][obj]['name'] not in obj_name:
                obj_name.append(sg[image_idx]['objects'][obj]['name'])
                for rel in sg[image_idx]['objects'][obj]['relations']:
                    if rel['name'] not in rel_name:
                        rel_name.append(rel['name'])
        node_name=obj_name+rel_name
        for node in node_name:
            for triplet in sg_concept[node]:
                # print(triplet)
                concept_file.write(triplet+'\n')

        os.system('CUDA_VISIBLE_DEVICES=0,1,2,3 python src/run_kbc_subgraph.py --dataset conceptnet --sim_relations --bert_concat --use_bias --load_model model/conceptnet_best_subgraph_model.pth --eval_only --write_results')
        os.system(f'mv concept_embed.pkl concept_sg_embed/{image_idx}.pkl')

def concept():

    node_names_file = open('cn_node_names.txt', 'r')
    node_names = [l.replace('\n', '') for l in node_names_file.readlines()]
    rel_file = open('rel.txt', 'r')
    rel_names = [l.replace('\n', '') for l in rel_file.readlines()]

    delay=1.5
    lim = 1000
    concept_file = open(f'concept_sg.json', 'w')
    node_sg_file = open('sg_node.txt', 'r')
    concept_sg={}
    for obj_name in node_sg_file.readlines():
        obj_name=obj_name.replace('\n', '')
        node_concept_list=[]
        offset = 0
        count=0
        while True:
            try:
                obj = requests.get('http://api.conceptnet.io/c/en/' + str(obj_name) + f'?offset={offset}&limit={lim}').json()
                time.sleep(delay)
                break
            except:
                if count>5:
                    break
                count+=1
                time.sleep(10*count)
        while 'edges' in obj and len(obj["edges"]) > 0:
            # print(f'{obj_name}: {len(obj["edges"])}')
            for e in obj['edges']:
                if 'language' in e['start'] and e['start']['language'] == 'en' and 'language' in e['end'] and e['end'][
                    'language'] == 'en':
                    start = clean(e['start']['label'])
                    end = clean(e['end']['label'])
                    rel = e['rel']['label']
                    if start in node_names and end in node_names and rel.lower() in rel_names:
                        line = rel + '\t' + start + '\t' + end
                        try:
                            line.encode('latin1')
                            node_concept_list.append(line)

                        except:
                            print(f'utf-8 error found: {line}')
            offset += lim
            count = 0
            while True:
                try:
                    obj = requests.get(
                        'http://api.conceptnet.io/c/en/' + str(obj_name) + f'?offset={offset}&limit={lim}').json()
                    time.sleep(delay)
                    break
                except:
                    if count > 5:
                        break
                    count += 1
                    time.sleep(10 * count)
        concept_sg[obj_name] = node_concept_list
    json.dump(concept_sg, concept_file)
    concept_file.close()


if __name__ == "__main__":
    # init()
    # concept()
    # sg_node_name()
    img_object_node_name()



