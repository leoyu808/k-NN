/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.io.IOException;
import java.util.List;
import java.util.LinkedList;
import java.util.concurrent.locks.ReentrantLock;

/**
 * This iterator iterates filterIdsArray to score if filter is provided else it iterates over all docs.
 * However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 */
public class NestedVectorIdsKNNIterator implements KNNIterator {
    protected final DocIdSetIterator filterIdsIterator;
    protected final float[] queryVector;
    private final byte[] quantizedQueryVector;
    protected final KNNFloatVectorValues knnFloatVectorValues;
    protected final SpaceType spaceType;
    protected int docId;
    private final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo;
    private final BitSet parentBitSet;
    private final ReentrantLock lock;

    protected final ThreadLocal<List<DocVector>> docVectors = ThreadLocal.withInitial(LinkedList::new);
    protected final ThreadLocal<Float> currentScore = ThreadLocal.withInitial(() -> Float.NEGATIVE_INFINITY);

    public NestedVectorIdsKNNIterator(
            @Nullable final DocIdSetIterator filterIdsIterator,
            final float[] queryVector,
            final KNNFloatVectorValues knnFloatVectorValues,
            final SpaceType spaceType,
            final BitSet parentBitSet
    ) throws IOException {
        this(filterIdsIterator, queryVector, knnFloatVectorValues, spaceType, parentBitSet, null, null);
    }

    public NestedVectorIdsKNNIterator(
            final float[] queryVector,
            final KNNFloatVectorValues knnFloatVectorValues,
            final SpaceType spaceType,
            final BitSet parentBitSet
    ) throws IOException {
        this(null, queryVector, knnFloatVectorValues, spaceType, parentBitSet, null, null);
    }

    public NestedVectorIdsKNNIterator (
            @Nullable final DocIdSetIterator filterIdsIterator,
            final float[] queryVector,
            final KNNFloatVectorValues knnFloatVectorValues,
            final SpaceType spaceType,
            final BitSet parentBitSet,
            final byte[] quantizedVector,
            final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) throws IOException {
        this.filterIdsIterator = filterIdsIterator;
        this.queryVector = queryVector;
        this.knnFloatVectorValues = knnFloatVectorValues;
        this.spaceType = spaceType;
        this.quantizedQueryVector = quantizedVector;
        this.segmentLevelQuantizationInfo = segmentLevelQuantizationInfo;
        this.parentBitSet = parentBitSet;
        this.docId = (filterIdsIterator == null) ? knnFloatVectorValues.nextDoc() : this.filterIdsIterator.nextDoc();
        this.lock = new ReentrantLock();
    }

    /**
     * Advance to the next best child doc per parent and update score with the best score among child docs from the parent.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next best child doc id
     */
    @Override
    public int nextDoc() throws IOException {
        lock.lock();
        try {
            if (docId == DocIdSetIterator.NO_MORE_DOCS) {
                return DocIdSetIterator.NO_MORE_DOCS;
            }
            int currentParent = parentBitSet.nextSetBit(docId);
            // In order to traverse all children for given parent, we have to use docId < parentId, because,
            // kNNVectorValues will not have parent id since DocId is unique per segment. For ex: let's say for doc id 1, there is one child
            // and for doc id 5, there are three children. In that case knnVectorValues iterator will have [0, 2, 3, 4]
            // and parentBitSet will have [1,5]
            // Hence, we have to iterate till docId from knnVectorValues is less than parentId instead of till equal to parentId
            while (docId != DocIdSetIterator.NO_MORE_DOCS && docId < currentParent) {
                knnFloatVectorValues.advance(docId);
                docVectors.get().add(new DocVector(docId, knnFloatVectorValues.getVector()));
                docId = (filterIdsIterator == null) ? knnFloatVectorValues.nextDoc() : this.filterIdsIterator.nextDoc();
            }
        } finally {
            lock.unlock();
        }

        int bestChild = -1;
        float bestScore = Float.NEGATIVE_INFINITY;
        while (!docVectors.get().isEmpty()) {
            DocVector dv = docVectors.get().removeFirst();
            float score = computeScore(dv.getVector());
            if (score > bestScore) {
                bestChild = dv.getDocId();
                bestScore = score;
            }
        }
        currentScore.set(bestScore);
        return bestChild;
    }
    
    private float computeScore(float[] vector) {
        if (segmentLevelQuantizationInfo != null && quantizedQueryVector != null) {
            byte[] quantizedVector = SegmentLevelQuantizationUtil.quantizeVector(vector, segmentLevelQuantizationInfo);
            return SpaceType.HAMMING.getKnnVectorSimilarityFunction().compare(quantizedQueryVector, quantizedVector);
        } else {
            // Calculates a similarity score between the two vectors with a specified function. Higher similarity
            // scores correspond to closer vectors.
            return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector);
        }
    }

    public float score() {
        return currentScore.get();
    }
}

