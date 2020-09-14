# cat🐈

This is the repository for the ACL 2020 paper [Embarrassingly Simple Unsupervised Aspect Extraction](https://www.aclweb.org/anthology/2020.acl-main.290/).
In this work, we extract aspects from restaurant reviews with attention that uses RBF kernels.

## Authors

* Stéphan Tulkens ([stephantul](https://www.github.com/stephantul))
* Andreas van Cranenburgh ([andreasvc](https://www.github.com/andreasvc))

## Requirements

* numpy
* gensim (for training embeddings)
* sklearn
* reach (for reading embeddings and vectorizing sentences)
* pyconll (for reading conll files)
* tqdm
* pandas
* matplotlib

Install these with `pip install -r requirements.txt`

## Using

If you want to apply cat to your data, you need a couple of things.

1. An aspect set, i.e., the set of labels you would like to predict.
2. A set of _in-domain_ word embeddings. This is really important, as we show in the paper.
3. A set of aspect terms which you think correspond to the aspects you want to extract. These do not need to be grouped by their aspect.
4. A set of instances for which you want to predict the labels you define in step 1. We expect these to be tokenized, one sentence per line.

If you have all these things, you can simply look at `example_pipeline/run.py` and replace the paths in this file with the paths to the appropriate files/instances.
cat🐈 has two hyperparameters: the gamma of the kernel, and the set of aspect words on which the attention is computed.

If you do not have access to pre-trained embeddings or aspect words, but you do have access to in-domain text, you will need a parser to extract either nouns or tree fragments.
For maximum portability, we adopt the CoNLLu format, a format that many parsers output.
If you use [spacy](https://spacy.io/), you can use the [spacyconllu](https://github.com/andreasvc/spacyconllu) script to convert text to CoNLLu format.

To obtain the nouns and embeddings for a given set of text in CoNLLu format, run `example_pipeline/preprocessing.py`, and replace the paths with the appropriate paths to your CoNLLu parsed file.
This will train your embeddings and extract the aspect words, which you can then use in `example_pipeline/run.py`.

## Adapting

If you just want to use or adapt `cat🐈` in your own project, check out `cat/simple.py`. This contains all the relevant code for computing the attention distribution.

## Reproducing

You can reproduce the experiments by obtaining the data, putting it in the `data/` folder and running the experiments from `experiments/`.
In the paper, we use the SemEval 2014, 2015 and citysearch dataset, which you can do here:

* [semeval 2014](http://alt.qcri.org/semeval2014/task4/)
* [semeval 2015](http://alt.qcri.org/semeval2015/task12/)
* [citysearch](https://www.cs.cmu.edu/~mehrbod/RR/) (we used the link from [this repository](https://github.com/ruidan/Unsupervised-Aspect-Extraction) because this link seems to be broken)

If you extract the text from these XML files and put the tokenized training data in `data/`, you can rerun our experiments.

## Citing

If you use the code or the techniques therein, please cite the paper:

```bibtex
@inproceedings{tulkens2020embarrassingly,
    title = "Embarrassingly Simple Unsupervised Aspect Extraction",
    author = "Tulkens, St{\'e}phan  and  van Cranenburgh, Andreas",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.290",
    doi = "10.18653/v1/2020.acl-main.290",
    pages = "3182--3187",
}
```

## License

GPL-V3
