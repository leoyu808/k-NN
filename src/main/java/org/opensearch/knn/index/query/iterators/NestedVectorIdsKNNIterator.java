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
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.io.IOException;

/**
 * This iterator iterates filterIdsArray to score if filter is provided else it iterates over all docs.
 * However, it dedupe docs per each parent doc
 * of which ID is set in parentBitSet and only return best child doc with the highest score.
 */
public class NestedVectorIdsKNNIterator extends VectorIdsKNNIterator {
    private final BitSet parentBitSet;

    public NestedVectorIdsKNNIterator(
            @Nullable final DocIdSetIterator filterIdsIterator,
            final float[] queryVector,
            final KNNFloatVectorValues knnFloatVectorValues,
            final SpaceType spaceType,
            final BitSet parentBitSet
    ) {
        this(filterIdsIterator, queryVector, knnFloatVectorValues, spaceType, parentBitSet, null, null);
    }

    public NestedVectorIdsKNNIterator(
            final float[] queryVector,
            final KNNFloatVectorValues knnFloatVectorValues,
            final SpaceType spaceType,
            final BitSet parentBitSet
    ) {
        this(null, queryVector, knnFloatVectorValues, spaceType, parentBitSet, null, null);
    }

    public NestedVectorIdsKNNIterator(
            @Nullable final DocIdSetIterator filterIdsIterator,
            final float[] queryVector,
            final KNNFloatVectorValues knnFloatVectorValues,
            final SpaceType spaceType,
            final BitSet parentBitSet,
            final byte[] quantizedVector,
            final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) {
        super(filterIdsIterator, queryVector, knnFloatVectorValues, spaceType, quantizedVector, segmentLevelQuantizationInfo);
        this.parentBitSet = parentBitSet;
    }

    /**
     * Advance to the next best child doc per parent and update score with the best score among child docs from the parent.
     * DocIdSetIterator.NO_MORE_DOCS is returned when there is no more docs
     *
     * @return next best child doc id
     */
    @Override
    public synchronized int nextDoc() throws IOException {
        return 0;
    }
}

