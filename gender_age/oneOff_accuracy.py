import pickle
import numpy as np

# Helper function which is used to implement OneOff accuracy
def buildOneOffMappings(le):
    # sort the class labels in ascending order (according to age) and initialize the one off mappings for computing accuracy
    classes = sorted(le.classes_, key=lambda x: int(x.split("_")[0]))
    oneOff = {}

    # loop over the index and name of the (sorted) class labels
    for (i, name) in enumerate(classes):
        # determine the index of the *current* class label name
        # in the *label encoder* (unordered) list, then
        # initialize the index of the previous and next age
        # groups adjacent to the current label
        name = str(name)
        current = np.where(le.classes_ == name)[0][0]
        prev = -1
        Next = -1

        # check to see if we should compute previous adjacent
        # age group
        if i > 0:
            prev = np.where(le.classes_ == classes[i - 1])[0][0]
        # check to see if we should compute the next adjacent
        # age group
        if i < len(classes) - 1:
            Next = np.where(le.classes_ == classes[i + 1])[0][0]

        # construct a tuple that consists of the current age
        # bracket, the previous age bracket, and the next age
        # bracket
        oneOff[current] = (current, prev, Next)

    # return the one-off mappings
    return oneOff

'''
>>> from agegenderhelper import buildOneOffMappings
>>> import pickle
>>> le = pickle.loads(open("/home/hrushikesh/dl4cv/gender_age/output/age_le.cpickle", "rb").read())
>>> buildOneOffMappings(le)
{0: (0, -1, 5), 5: (5, 0, 7), 7: (7, 5, 1), 1: (1, 7, 2), 2: (2, 1, 3), 3: (3, 2, 4), 4: (4, 3, 6), 6: (6, 4, -1)}
'''

