import os
from rdflib import Graph, URIRef, Namespace
from collections import defaultdict
import sys
from visualize_bbox import draw_bbox
import ujson as json


# prefix_AIDA = 'https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/InterchangeOntology#'
# AIDA = Namespace(prefix_AIDA)
# RDF = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
AIDA = Namespace('https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#')
RDF = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')


def get_str(uri):
    return uri.toPython().split('#')[-1]

def load_objects(turtle_content):
    if 'ObjectDetection' not in turtle_content:
        return None, None

    g = Graph().parse(data=turtle_content, format='ttl')

    bbox_dict = defaultdict(lambda : defaultdict())
    offset_dict = defaultdict(lambda : defaultdict())
    for s, p, o in g:
        p = get_str(p)
        if 'boundingBoxLowerRightX' == p or 'boundingBoxLowerRightY' == p \
                or 'boundingBoxUpperLeftX' == p or 'boundingBoxUpperLeftY' == p:
            bbox_dict[s][p] = o
        elif 'boundingBox' == p or 'source' == p:
            offset_dict[s][p] = o
    # print(offset_dict)

    offset_bnode = dict()
    bnode_offset = dict()
    for one_bnode in offset_dict:
        if 'boundingBox' not in offset_dict[one_bnode]:
            # print('Not image offset', one_bnode)
            continue
        bbox = offset_dict[one_bnode]['boundingBox']
        docid = offset_dict[one_bnode]['source'].toPython()
        print(offset_dict[one_bnode])
        print(docid, bbox_dict[bbox])
        for one_offset_type in bbox_dict[bbox]:
            if 'boundingBoxLowerRightX' == one_offset_type:
                lrx_offset = int(bbox_dict[bbox][one_offset_type])
            elif 'boundingBoxLowerRightY' == one_offset_type:
                lry_offset = int(bbox_dict[bbox][one_offset_type])
            elif 'boundingBoxUpperLeftX' == one_offset_type:
                ulx_offset = int(bbox_dict[bbox][one_offset_type])
            elif 'boundingBoxUpperLeftY' == one_offset_type:
                uly_offset = int(bbox_dict[bbox][one_offset_type])
        search_key = "%s:%d-%d-%d-%d" % (docid, lrx_offset, lry_offset, ulx_offset, uly_offset)
        # print(search_key)
        offset_bnode[search_key] = one_bnode
        bnode_offset[one_bnode] = search_key
    # print(bnode_offset)

    image_objects = defaultdict(set)
    object_info = defaultdict(lambda: defaultdict(str))
    for entity in g.subjects(predicate=RDF.type, object=AIDA.Entity):
        # get infomative justification
        info_justi = g.value(subject=entity, predicate=AIDA.informativeJustification)
        if info_justi not in bnode_offset:
            continue
        info_offset = bnode_offset[info_justi]
        imageid = info_offset.split(':')[0]
        object_info[entity]['infojusti'] = info_offset
        image_objects[imageid].add(entity)
        # get type
        for assertion in g.subjects(object=entity, predicate=RDF.subject):
            object_assrt = g.value(subject=assertion, predicate=RDF.object)
            predicate_assrt = g.value(subject=assertion, predicate=RDF.predicate)
            if predicate_assrt == RDF.type:
                object_info[entity]['type'] = get_str(object_assrt)

    image_objects_info = defaultdict(lambda : defaultdict(lambda : defaultdict()))
    for image in image_objects:
        for object in image_objects[image]:
            image_objects_info[image][object]['label'] = object_info[object]['type']
            image_objects_info[image][object]['infojusti'] = object_info[object]['infojusti']
    return image_objects_info

if __name__ == '__main__':
    input_ttl_folder = '/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/sample_one/m36_vision'
    output_folder = '/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/sample_one/vis/objects'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for one_file in os.listdir(input_ttl_folder):
        print(one_file)
        if not one_file.endswith(".ttl"):
            continue
        one_file_id = one_file.replace(".ttl", "")
        one_file_path = os.path.join(input_ttl_folder, one_file)
        output_file = os.path.join(output_folder, one_file_id+'.json')
        turtle_content = open(one_file_path).read()

        image_objects_info = load_objects(turtle_content)

        json.dumps(image_objects_info, open(output_file, 'w'), indent=4)