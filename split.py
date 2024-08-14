import splitfolders

splitfolders.ratio("dataset/train-set", # The location of dataset
                   output="surfaces", # The output location
                   seed=42, # The number of seed
                   ratio=(.8, 0, .2), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )