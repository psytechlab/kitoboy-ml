"""Script for executing the requets to [NLP Parser](https://github.com/IINemo/isanlp)"""
from isanlp import PipelineCommon
from isanlp.processor_remote import ProcessorRemote
from isanlp.processor_razdel import ProcessorRazdel
import csv
import time




def extr_pairs(tree, text):
    pp = []
    if tree.left is None and tree.right is None:
        # If both left and right nodes are None, add the current text segment
        pp.append(tree.text.strip())
    else:
        # Recurse left first, then right
        if tree.left:
            pp += extr_pairs(tree.left, text)
        if tree.right:
            pp += extr_pairs(tree.right, text)
    return pp


if __name__ == "__main__":

    # put the address here ->
    address_syntax = ('127.0.1.1', 3334)
    address_rst = ('127.0.1.1', 3335)

    ppl_ru = PipelineCommon([
        (ProcessorRazdel(), ['text'],
        {'tokens': 'tokens',
        'sentences': 'sentences'}),
        (ProcessorRemote(address_syntax[0], address_syntax[1], '0'),
        ['tokens', 'sentences'],
        {'lemma': 'lemma',
        'morph': 'morph',
        'syntax_dep_tree': 'syntax_dep_tree',
        'postag': 'postag'}),
        (ProcessorRemote(address_rst[0], address_rst[1], 'default'),
        ['text', 'tokens', 'sentences', 'postag', 'morph', 'lemma', 'syntax_dep_tree'],
        {'rst': 'rst'})
    ])

    with open("suicide_examples.txt", "r", encoding="utf-8") as r_f:
        texts = [s.strip() for s in r_f if s.strip() != ""]


    with open("result_isanlp.csv", "w", encoding="utf-8") as w_f:
        file_writer = csv.writer(w_f, delimiter=",", lineterminator="\n")
        start_time = time.time()
        for text in texts:
            res = ppl_ru(text)
            new_res = extr_pairs(res['rst'][0], res['text'])
            file_writer.writerow([text, new_res])
            print(new_res)
        end_time = time.time()

    execution_time = end_time - start_time
    print(f'Execution time: {execution_time} seconds')
