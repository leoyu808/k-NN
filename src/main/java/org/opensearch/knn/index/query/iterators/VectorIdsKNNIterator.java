/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.io.IOException;

/**
 * Inspired by DiversifyingChildrenFloatKnnVectorQuery in lucene
 * https://github.com/apache/lucene/blob/7b8aece125aabff2823626d5b939abf4747f63a7/lucene/join/src/java/org/apache/lucene/search/join/DiversifyingChildrenFloatKnnVectorQuery.java#L162
 *
 * The class is used in KNNWeight to score all docs, but, it iterates over filterIdsArray if filter is provided
 */
public class VectorIdsKNNIterator implements KNNIterator {
    protected final DocIdSetIterator filterIdsIterator;
    protected final float[] queryVector;
    private final byte[] quantizedQueryVector;
    protected final KNNFloatVectorValues knnFloatVectorValues;
    protected final SpaceType spaceType;
    protected float currentScore = Float.NEGATIVE_INFINITY;
    protected int minDocId;
    protected int maxDocId;
    protected int docId;
    private final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo;

    public VectorIdsKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final int minDocId,
        final int maxDocId
    ) throws IOException {
        this(filterIdsIterator, queryVector, knnFloatVectorValues, spaceType, null, null,  minDocId, maxDocId);
    }

    public VectorIdsKNNIterator(final float[] queryVector, final KNNFloatVectorValues knnFloatVectorValues, final SpaceType spaceType, final int minDocId, final int maxDocId)
        throws IOException {
        this(null, queryVector, knnFloatVectorValues, spaceType, null, null, minDocId, maxDocId);
    }

    public VectorIdsKNNIterator(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final float[] queryVector,
        final KNNFloatVectorValues knnFloatVectorValues,
        final SpaceType spaceType,
        final byte[] quantizedQueryVector,
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo,
        final int minDocId,
        final int maxDocId
    ) throws IOException {
        this.filterIdsIterator = filterIdsIterator;
        this.queryVector = queryVector;
        this.knnFloatVectorValues = knnFloatVectorValues;
        this.spaceType = spaceType;
        this.minDocId = minDocId;
        this.maxDocId = maxDocId;
        // This cannot be moved inside nextDoc() method since it will break when we have nested field, where
        // nextDoc should already be referring to next knnVectorValues
        int firstDoc;
        if (filterIdsIterator != null) {
            firstDoc = filterIdsIterator.advance(minDocId);
            if (firstDoc != DocIdSetIterator.NO_MORE_DOCS) {
                knnFloatVectorValues.advance(firstDoc);
            }
        } else {
            firstDoc = knnFloatVectorValues.advance(minDocId);
        }
        if (firstDoc >= maxDocId) {
            this.docId = DocIdSetIterator.NO_MORE_DOCS;
        } else {
            this.docId = firstDoc;
        }

        this.quantizedQueryVector = quantizedQueryVector;
        this.segmentLevelQuantizationInfo = segmentLevelQuantizationInfo;
    }

    /**
     * Advance to the next doc and update score value with score of the next doc.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next doc id
     */
    @Override
    public int nextDoc() throws IOException {
        if (docId == DocIdSetIterator.NO_MORE_DOCS) {
            return DocIdSetIterator.NO_MORE_DOCS;
        }
        currentScore = computeScore();
        int currentDocId = docId;
        docId = getNextDocId();
        if (docId >= maxDocId) {
            docId = DocIdSetIterator.NO_MORE_DOCS;
        }
        return currentDocId;
    }

    @Override
    public float score() {
        return currentScore;
    }

    protected float computeScore() throws IOException {
        final float[] vector = knnFloatVectorValues.getVector();
        if (segmentLevelQuantizationInfo != null && quantizedQueryVector != null) {
            byte[] quantizedVector = SegmentLevelQuantizationUtil.quantizeVector(vector, segmentLevelQuantizationInfo);
            return SpaceType.HAMMING.getKnnVectorSimilarityFunction().compare(quantizedQueryVector, quantizedVector);
        } else {
            // Calculates a similarity score between the two vectors with a specified function. Higher similarity
            // scores correspond to closer vectors.
            return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vector);
        }
    }

    protected int getNextDocId() throws IOException {
        if (filterIdsIterator == null) {
            return knnFloatVectorValues.nextDoc();
        }
        int nextDocID = this.filterIdsIterator.nextDoc();
        // For filter case, advance vector values to corresponding doc id from filter bit set
        if (nextDocID != DocIdSetIterator.NO_MORE_DOCS) {
            knnFloatVectorValues.advance(nextDocID);
        }
        return nextDocID;
    }
}
