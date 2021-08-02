import xml.etree.ElementTree as ET
import os


def parse_offset_str(offset_str):
    docid = offset_str[:offset_str.rfind(':')]
    start = int(offset_str[offset_str.rfind(':') + 1:offset_str.rfind('-')])
    end = int(offset_str[offset_str.rfind('-') + 1:])
    return docid, start, end

class LTF_util(object):
    def __init__(self, ltf_dir):
        super(LTF_util, self).__init__()
        self.ltf_dir = ltf_dir

    def parse_offset_str(self, offset_str):
        return parse_offset_str(offset_str)

    def get_context(self, offset_str):
        docid, start, end = self.parse_offset_str(offset_str)

        tokens = []

        ltf_file_path = os.path.join(self.ltf_dir, docid + '.ltf.xml')
        if not os.path.exists(ltf_file_path):
            return '[ERROR]NoLTF'
        tree = ET.parse(ltf_file_path)
        root = tree.getroot()
        for doc in root:
            for text in doc:
                for seg in text:
                    seg_beg = int(seg.attrib["start_char"])
                    seg_end = int(seg.attrib["end_char"])
                    if start >= seg_beg and end <= seg_end:
                        for token in seg:
                            if token.tag == "TOKEN":
                                tokens.append(token.text)
                    if len(tokens) > 0:
                        return tokens
        return tokens

    def get_context_html(self, offset_str):
        docid, start, end = self.parse_offset_str(offset_str)

        tokens = []

        ltf_file_path = os.path.join(self.ltf_dir, docid + '.ltf.xml')
        if not os.path.exists(ltf_file_path):
            return '[ERROR]NoLTF'
        tree = ET.parse(ltf_file_path)
        root = tree.getroot()
        for doc in root:
            for text in doc:
                for seg in text:
                    seg_beg = int(seg.attrib["start_char"])
                    seg_end = int(seg.attrib["end_char"])
                    if start >= seg_beg and end <= seg_end:
                        for token in seg:
                            if token.tag == "TOKEN":
                                token_text = token.text
                                token_beg = int(token.attrib["start_char"])
                                token_end = int(token.attrib["end_char"])
                                if start <= token_beg and end >= token_end:
                                    token_text = '<span style="color:blue">' + token_text + '</span>'
                                tokens.append(token_text)
                    if len(tokens) > 0:
                        return ' '.join(tokens)
        return '[ERROR]'


    def get_str(self, offset_str):
        docid, start, end = self.parse_offset_str(offset_str)

        tokens = []

        ltf_file_path = os.path.join(self.ltf_dir, docid + '.ltf.xml')
        if not os.path.exists(ltf_file_path):
            return '[ERROR]NoLTF'
        tree = ET.parse(ltf_file_path)
        root = tree.getroot()
        for doc in root:
            for text in doc:
                for seg in text:
                    for token in seg:
                        if token.tag == "TOKEN":
                            # print(token.attrib["start_char"])
                            token_beg = int(token.attrib["start_char"])
                            token_end = int(token.attrib["end_char"])
                            if start <= token_beg and end >= token_end:
                                tokens.append(token.text)
        if len(tokens) > 0:
            return ' '.join(tokens)
        # Todo: uncomment
#         print('[ERROR]can not find the string with offset ', offset_str)
        return None

    def get_str_inside_sent(self, offset_str):
        docid, start, end = self.parse_offset_str(offset_str)

        tokens = []

        ltf_file_path = os.path.join(self.ltf_dir, docid + '.ltf.xml')
        if not os.path.exists(ltf_file_path):
            return '[ERROR]NoLTF'
        tree = ET.parse(ltf_file_path)
        root = tree.getroot()
        for doc in root:
            for text in doc:
                for seg in text:
                    seg_beg = int(seg.attrib["start_char"])
                    seg_end = int(seg.attrib["end_char"])
                    if start >= seg_beg and end <= seg_end:
                        for token in seg:
                            if token.tag == "TOKEN":
                                # print(token.attrib["start_char"])
                                token_beg = int(token.attrib["start_char"])
                                token_end = int(token.attrib["end_char"])
                                if start <= token_beg and end >= token_end:
                                    tokens.append(token.text)
                    if len(tokens) > 0:
                        return ' '.join(tokens)
        print('[ERROR]can not find the string with offset ', offset_str)
        return None


if __name__ == '__main__':
    ltf_dir = '/data/m1/lim22/aida2019/dryrun/source/ru'
    ltf_util = LTF_util(ltf_dir)
    print(ltf_util.get_context('HC000Q7NP:167-285'))