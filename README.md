# Rainforest Connection Species Audio Detection

> Given animal audio clip, correctly classify them

A post-competition organization of the code used in the competition. Because i find the resources provided by Kaggle (30h free TPU) to be sufficient, most of the code are written in the kernel notebook in kaggle platform (although i am not really a fan of using it). I started using the baseline code written by yosshi999 and made changes along the way, implementing my own model architectures.

# What's new?

This solution places at top 9%. However, I did manage to have a single model that reaches  `0.891`. More diversification might result in a better score. My implementations includes some of the following:

1) random resized crop as augmentation
2) custom channel attention layer + spatial attention layer + CNN blocks inspired by a winner from a similar competition. The model design can be found in `rainforest_audio/model.py`

# Run 

Run all 7 densenet notebook and generate the data. Place them in their respective data folder. Also, get one of a public [notebook](https://www.kaggle.com/mehrankazeminia/ensembling-0-880-audio-detection-101). All the csv are uploaded in the `/data/` folder. 

To generate the submission, run:

```
$ make run_submission
```

# Post-competition

Mask-loss did not worked for me. Have yet to try re-training all my models with Lsoft objective function. Thing that did not work are placed in `archive` folder within the notebook.

I have also added a summary of learning points and experience in a pdf document based on an internal sharing. It is found in `Rainforest Audio Detection.pdf`
