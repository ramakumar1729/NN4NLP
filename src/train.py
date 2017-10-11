import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

import models

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def train():
    training_data = [
                   (
                     ("Traditional <C> <entity> information retrieval techniques </entity>"
                      "use a"
                      " <entity> histogram </entity> "
                      "of"
                      " <P> <entity> keywords </entity> "
                      "as the"
                      " <entity> document representation </entity> "
                      "but"
                      " <entity> oral communication </entity> "
                      "may offer additional"
                      " <entity> indices </entity> "
                      "such as the time and place of the rejoinder and the attendance.").split(), ["USAGE"]),
                   (
                     ("Traditional <P> <entity> information retrieval techniques </entity>"
                      "use a"
                      " <entity> histogram </entity> "
                      "of"
                      " <C> <entity> keywords </entity> "
                      "as the"
                      " <entity> document representation </entity> "
                      "but"
                      " <entity> oral communication </entity> "
                      "may offer additional"
                      " <entity> indices </entity> "
                      "such as the time and place of the rejoinder and the attendance.").split(), ["NONE"]),
                   ]

    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)
    tag_to_ix = {"USAGE": 0, "NONE":  1}
    ix_to_tag = {0: "USAGE", 1: "NONE"}
    
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100
    
    model = models.NRE(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
    
    for epoch in range(300):  
        for sentence, tags in training_data:
            model.zero_grad()
    
            model.hidden = model.init_hidden()
    
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
    
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
    
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print("Tag for sentence:")
    print(training_data[0][0])
    idx = tag_scores.topk(1)[1].data.numpy()[0][0]
    print(ix_to_tag[idx])

if __name__ == '__main__':
        train()
