import os
from rdflib import Graph, Namespace, URIRef
from collections import defaultdict
import sys
from ltf_util import LTF_util
from visualize_bbox import draw_bbox
import json # import ujson as json

data_dir = '/shared/nas/data/m1/manling2/aida_docker_test/uiuc_ie_pipeline_fine_grained/data/sample_data/VOA_EN_NW_2017_sample50'
output_folder = os.path.join(data_dir, "vis") 

input_ttl_folder = os.path.join(data_dir, "cu_graph_merging_ttl/merged_ttl")
ltf_dir = os.path.join(data_dir, "ltf")
rsd_dir = os.path.join(data_dir, "rsd")
images_path = os.path.join(data_dir, "vision/data/jpg/jpg")
visualpath = os.path.join(output_folder, "grounding_visualization")
resultpath = os.path.join(output_folder, "grounding_result")
patch_visual_path = 'patchs'
patch_path = os.path.join(visualpath, patch_visual_path)

# AIDA = Namespace('https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/InterchangeOntology#')
AIDA = Namespace('https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#')
RDF = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
# LDC = Namespace(prefix_ldc[eval])

ltf_util = LTF_util(ltf_dir)

if not os.path.exists(resultpath):
    os.makedirs(resultpath)
if not os.path.exists(visualpath):
    os.makedirs(visualpath)
if not os.path.exists(patch_path):
    os.makedirs(patch_path)
record_count = 0
page_limit = 100


head = '''
    <!DOCTYPE html>
    <html>
    <head>
    <title>Page Title</title>
    </head>
    <body>
    '''
tail = '''
    </body>
    </html>
    '''

# writer = open(resultpath, 'w')
for one_file in os.listdir(input_ttl_folder):
    # print(one_file)
    if not one_file.endswith(".ttl"):
        continue
    one_file_id = one_file.replace(".ttl", "")
    one_file_path = os.path.join(input_ttl_folder, one_file)
    output_file = os.path.join(output_folder, one_file)
    turtle_content = open(one_file_path).read()

    if 'GroundingBox' not in turtle_content:
        continue

    # f_html = open(os.path.join(visualpath, 'grounding_%d.html' % int(record_count/page_limit)), 'a+')
    html_file_path = os.path.join(visualpath, '%s.html' % one_file_id)
    with open(html_file_path, 'w') as writer:
        writer.write(head)
        writer.write('<b>[Raw Text]</b>: %s\n<br>' % (open(os.path.join(rsd_dir, one_file_id+'.rsd.txt')).read()))
        writer.write('======================================================================\n<br><br><br>')
    f_html = open(html_file_path, 'a+')

    g = Graph().parse(data=turtle_content, format='ttl')

    # get events, related entities
    # get all entities
    entities_text = []
    entities_grounding = []
    events = []
    clusters_grounding = []
    # args = []
    for s, p, o in g:
        # print(s, p, o)
        if 'type' in p:
            # if 'Entity' in o:
            #     if 'columbia' not in s:
            #         entities_text.append(s)
            #     elif 'GroundingBox' in s:
            #         entities_grounding.append(s)
            #     # else:
            #     #     print('unuseful entity: ', s)
            if 'Event' in o:
                events.append(s)
            elif 'SameAsCluster' in o and 'Grounding' in s:
                clusters_grounding.append(s)
    # print('entities text: ', len(entities_text))
    # print('entities grounding: ', len(entities_grounding))
    print('clusters grounding: ', len(clusters_grounding))
    print('events: ', len(events))

    if len(clusters_grounding) == 0:
        continue

    # get visual entities grounding to which text entity
    # <text_entity> -> <visual entity>
    text_visual = defaultdict(dict)
    text_events = defaultdict(list)
    # for entity_grounding in entities_grounding:
    #     cluster_state = g.value(predicate=AIDA.clusterMember, object=entity_grounding)
    #     cluster = g.value(subject=cluster_state, predicate=AIDA.cluster)
    #     for cluster_state in g.subjects(AIDA.cluster, cluster):
    #         for entity in g.objects(cluster_state, AIDA.clusterMember):
    #             if entity in entities_text:
    #                 text_visual[entity] = entity_grounding
    #     # for cluster_state in g.subjects(AIDA.clusterMember, entity_grounding):
    #     #     for cluster in g.objects(cluster_state, AIDA.cluster):
    #     #         for cluster_state in g.subjects(AIDA.cluster, cluster):
    #     #             for entity in g.objects(cluster_state, AIDA.clusterMember):
    #     #                 if entity in entities_text:
    #     #                     text_visual[entity] = entity_grounding
    for cluster in clusters_grounding:
        entity_text = g.value(subject=cluster, predicate=AIDA.prototype)

        # for event_state in g.subjects(predicate=RDF.object, object=entity_text):
        #     event = g.value(subject=event_state, predicate=RDF.subject)
        #     if event not in events:
        #         continue
        #     else:
        #         text_events[entity_text].append(event)

        for cluster_state in g.subjects(AIDA.cluster, cluster):
            for entity_grounding in g.objects(cluster_state, AIDA.clusterMember):
                if entity_text == entity_grounding:
                    continue
                confidence = g.value(subject=g.value(subject=cluster_state, predicate=AIDA.confidence), predicate=AIDA.confidenceValue)
                # print(entity_grounding, confidence)
                text_visual[entity_text][entity_grounding] = confidence

    # print(text_visual)

    result = defaultdict(lambda : defaultdict(list))
    for entity_text in text_visual:
        record_count = record_count + 1
        # entity from text
        f_html.write('<b>Mention</b>: %s\n<br>' % (entity_text))
        for entity_text_justi in g.objects(subject=entity_text, predicate=AIDA.justifiedBy):
            entity_text_doc = g.value(subject=entity_text_justi, predicate=AIDA.source)
            entity_text_start = g.value(subject=entity_text_justi, predicate=AIDA.startOffset)
            entity_text_end = g.value(subject=entity_text_justi, predicate=AIDA.endOffsetInclusive)
            entity_text_offset = '%s:%s-%s' % (entity_text_doc, entity_text_start, entity_text_end)
            # writer.write('%s\t' % (entity_text_offset))
            result[entity_text]['offset'].append(entity_text_offset)
            entity_label = ltf_util.get_str(entity_text_offset)
            context = ltf_util.get_context_html(entity_text_offset)
            f_html.write(' ---- %s\t%s\n<br><br>' % (entity_text_offset, context)) # entity_label
        # writer.write('\n')
        # visual objects
        f_html.write('Image: \n<br>')
        for entity_grounding in text_visual[entity_text]:
            confidence = text_visual[entity_text][entity_grounding]
            f_html.write('<b>GroundingConfidence</b>: %s, %s\n<br>' % (entity_grounding, confidence))
            for entity_image_justi in g.objects(subject=entity_grounding, predicate=AIDA.justifiedBy):
                entity_image_doc = g.value(subject=entity_image_justi, predicate=AIDA.source)
                if entity_image_doc.find('.') > -1:
                    entity_image_doc = entity_image_doc[:entity_image_doc.find('.')]
                entity_image_doc_filename = entity_image_doc + '.jpg'
                entity_image_bbox = g.value(subject=entity_image_justi, predicate=AIDA.boundingBox)
                entity_bbox_LRX = int(g.value(subject=entity_image_bbox, predicate=AIDA.boundingBoxLowerRightX))
                entity_bbox_LRY = int(g.value(subject=entity_image_bbox, predicate=AIDA.boundingBoxLowerRightY))
                entity_bbox_ULX = int(g.value(subject=entity_image_bbox, predicate=AIDA.boundingBoxUpperLeftX))
                entity_bbox_ULY = int(g.value(subject=entity_image_bbox, predicate=AIDA.boundingBoxUpperLeftY))
                # print(entity_bbox_ULX, entity_bbox_ULY, entity_bbox_LRX, entity_bbox_LRY)
                # image_file_path = os.path.join(images_path, entity_image_doc)
                # f_html.write("<img src = \""+image_file_path+"\" width=\"100\%\">\n<br>")
                target_file_name = '%s-%d-%d-%d-%d.png' % (
                            entity_image_doc, entity_bbox_ULX, entity_bbox_ULY, entity_bbox_LRX, entity_bbox_LRY)
                result[entity_text]['bboxs'].append(target_file_name)
                result[entity_text]['bboxs_confidence'].append(confidence)
                target_patch_file_path = os.path.join(patch_path, target_file_name)
                draw_bbox(images_path, entity_image_doc_filename, target_patch_file_path, entity_bbox_ULX, entity_bbox_ULY, entity_bbox_LRX, entity_bbox_LRY)
                f_html.write("<img src=\"" + os.path.join(patch_visual_path, target_file_name) + "\" width=\"300\">\n<br>")
        # writer.write('\n')
        # related events:
        f_html.write('Event: \n<br>')
        # for event in text_events[entity_text]:
        for event_state in g.subjects(predicate=RDF.object, object=entity_text):
            event = g.value(subject=event_state, predicate=RDF.subject)
            if event not in events:
                continue
            role = g.value(subject=event_state, predicate=RDF.predicate).split('#')[-1]  #.replace(prefix, "")
            # the justification of this role
            role_justi = g.value(subject=event_state, predicate=AIDA.justifiedBy)
            role_doc = g.value(subject=role_justi, predicate=AIDA.source)
            role_start = g.value(subject=role_justi, predicate=AIDA.startOffset)
            role_end = g.value(subject=role_justi, predicate=AIDA.endOffsetInclusive)
            # writer.write('%s|%s:%s-%s' % (role, role_doc, role_start, role_end))
            dict_tmp = dict()
            dict_tmp['event'] = list()
            dict_tmp['role'] = role
            dict_tmp['offset'] = '%s:%s-%s' % (role_doc, role_start, role_end)
            for event_text_justi in g.objects(subject=event, predicate=AIDA.justifiedBy):
                # event_text_doc = g.value(subject=event_text_justi, predicate=AIDA.source)
                event_text_start = g.value(subject=event_text_justi, predicate=AIDA.startOffset)
                event_text_end = g.value(subject=event_text_justi, predicate=AIDA.endOffsetInclusive)
                # if role_start <= event_text_start and event_text_end <= role_end:
                event_text_offset = '%s:%s-%s' % (role_doc, event_text_start, event_text_end)
                dict_tmp['event'].append(event_text_offset)
                event_label = ltf_util.get_str(event_text_offset)
                event_context = ltf_util.get_context_html(event_text_offset)
                f_html.write('%s\n<br>%s\t%s\t%s\n<br>' % (role, event_label, event_text_offset, event_context))
            result[entity_text]['roles'].append(dict_tmp)
        # writer.write('\n')
        json.dump(result, open(os.path.join(resultpath, one_file.replace('.ttl', '.json')), 'w'), indent=4)
        f_html.write('======================================================================\n<br><br><br>')

    f_html.write(tail)
    # writer.flush()
    f_html.flush()
    # writer.close()
    f_html.close()


# for html in os.listdir(visualpath):
#     html_content = open(os.path.join(visualpath, html)).read()
#     html_new = open(os.path.join(visualpath, html), 'w')
#     html_new.write('%s\n' % head)
#     html_new.write(html_content)
#     html_new.write('%s\n' % tail)
#     html_new.flush()
#     html_new.close()