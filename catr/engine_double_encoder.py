# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

import math
import sys
import tqdm
import nltk

from models import utils

def get_bleu_score(truths, predicteds, weights_to_be_used=[0.25, 0.25, 0.25, 0.25]):
    scores = []
    for index in range(len(truths)):
        truth = truths[index]
        predicted = predicteds[index]
        try:
            score = nltk.translate.bleu_score.sentence_bleu([truth], predicted, weights=weights_to_be_used)
        except:
            score = 0
        scores.append(score)
    return sum(scores)/len(scores)

def train_one_epoch_double_encoder(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm, word2ind):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    bleu_score = 0.0
    bleu_score_iteration_cutoff = 50
    total = len(data_loader)
    iteration_number = 0
    with tqdm.tqdm(total=total) as pbar:
        for image_features, caps, cap_masks in data_loader:
            iteration_number += 1
            # samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            
            images_chexpert = image_features[0].to(device)
            images_imagenet = image_features[1].to(device)
    
            outputs = model(images_chexpert, images_imagenet, caps[:, :-1], cap_masks[:, :-1])
            # getting average bleu score of iteration
            if iteration_number <= bleu_score_iteration_cutoff:
                outputs_pred = torch.argmax(outputs, dim=2).cpu()
                all_outputs = outputs_pred.numpy()
                modified_caps = caps[:,:-1].cpu()
                all_modified_caps = modified_caps.numpy()
                all_outputs_corrected = []
                for report in all_outputs:
                    if (report == word2ind["</s>"]).any():
                        end_index = (report == word2ind["</s>"]).nonzero()[0][0]
                        report = report[:end_index+1]
                    all_outputs_corrected.append(report)
                all_captions_corrected = []
                for report in all_modified_caps:
                    if (report == word2ind["</s>"]).any():
                        end_index = (report == word2ind["</s>"]).nonzero()[0][0]
                        report = report[:end_index+1]
                    all_captions_corrected.append(report)
                bleu_score_av = get_bleu_score(all_captions_corrected, all_outputs_corrected)
                bleu_score += bleu_score_av
            
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total, bleu_score / bleu_score_iteration_cutoff

@torch.no_grad()
def evaluate_double_encoder(model, criterion, data_loader, device, word2ind):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)
    bleu_score = 0.0

    with tqdm.tqdm(total=total) as pbar:
        for image_features, caps, cap_masks in data_loader:
            # samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            
            images_chexpert = image_features[0]
            images_imagenet = image_features[1]

            outputs = model(images_chexpert, images_imagenet, caps[:, :-1], cap_masks[:, :-1])
            # getting average bleu score of iteration
            outputs_pred = torch.argmax(outputs, dim=2).cpu()
            all_outputs = outputs_pred.numpy()
            modified_caps = caps[:,:-1].cpu()
            all_modified_caps = modified_caps.numpy()
            all_outputs_corrected = []
            for report in all_outputs:
                if (report == word2ind["</s>"]).any():
                    end_index = (report == word2ind["</s>"]).nonzero()[0][0]
                    report = report[:end_index+1]
                all_outputs_corrected.append(report)
            all_captions_corrected = []
            for report in all_modified_caps:
                if (report == word2ind["</s>"]).any():
                    end_index = (report == word2ind["</s>"]).nonzero()[0][0]
                    report = report[:end_index+1]
                all_captions_corrected.append(report)
            bleu_score_av = get_bleu_score(all_captions_corrected, all_outputs_corrected)
            bleu_score += bleu_score_av
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            validation_loss += loss.item()

            pbar.update(1)
        
    return validation_loss / total, bleu_score / total