package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.SegmentLevelQuantizationInfo;
import org.opensearch.knn.index.query.SegmentLevelQuantizationUtil;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;

import java.io.IOException;

/**
 * Thread‑safe iterator over KNN vector docs: multiple threads can call nextDoc()
 * and each doc will be returned exactly once across all threads. Score() and
 * docId() are maintained per‐thread.
 */
public class VectorIdsKNNIterator implements KNNIterator {
    private final Object lock = new Object();
    private final DocIdSetIterator filterIdsIterator;
    private final float[] queryVector;
    private final byte[] quantizedQueryVector;
    private final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo;
    private final KNNFloatVectorValues knnFloatVectorValues;
    private final SpaceType spaceType;

    protected final ThreadLocal<Integer> docId = ThreadLocal.withInitial(() -> -1);
    protected final ThreadLocal<Float>   currentScore = ThreadLocal.withInitial(() -> Float.NEGATIVE_INFINITY);

    public VectorIdsKNNIterator(
            @Nullable DocIdSetIterator filterIdsIterator,
            float[] queryVector,
            KNNFloatVectorValues knnFloatVectorValues,
            SpaceType spaceType
    ) {
        this(filterIdsIterator, queryVector, knnFloatVectorValues, spaceType, null, null);
    }

    public VectorIdsKNNIterator(
            float[] queryVector,
            KNNFloatVectorValues knnFloatVectorValues,
            SpaceType spaceType
    ) {
        this(null, queryVector, knnFloatVectorValues, spaceType, null, null);
    }

    public VectorIdsKNNIterator(
            @Nullable DocIdSetIterator filterIdsIterator,
            float[] queryVector,
            KNNFloatVectorValues knnFloatVectorValues,
            SpaceType spaceType,
            byte[] quantizedQueryVector,
            SegmentLevelQuantizationInfo segmentLevelQuantizationInfo
    ) {
        this.filterIdsIterator = filterIdsIterator;
        this.queryVector = queryVector;
        this.knnFloatVectorValues = knnFloatVectorValues;
        this.spaceType = spaceType;
        this.quantizedQueryVector = quantizedQueryVector;
        this.segmentLevelQuantizationInfo = segmentLevelQuantizationInfo;
    }

    /**
     * Advance to the next doc and compute its score. Returns
     * DocIdSetIterator.NO_MORE_DOCS when finished.
     */
    @Override
    public int nextDoc() throws IOException {
        int doc;
        float[] vectorCopy;

        // 1) Under lock, pull the next doc, advance the vector reader, and copy the vector.
        synchronized (lock) {
            doc = getNextDocId();
            if (doc == DocIdSetIterator.NO_MORE_DOCS) {
                docId.set(DocIdSetIterator.NO_MORE_DOCS);
                currentScore.set(Float.NEGATIVE_INFINITY);
                return DocIdSetIterator.NO_MORE_DOCS;
            }
            // grab a copy of the underlying vector so we can score it outside the lock
            float[] raw = knnFloatVectorValues.getVector();
            vectorCopy = raw.clone();
        }

        // 3) Publish to thread‑local state
        docId.set(doc);
        currentScore.set(computeScore(vectorCopy));
        return doc;
    }

    protected float computeScore(float[] vectorCopy) throws IOException {
        if (segmentLevelQuantizationInfo != null && quantizedQueryVector != null) {
            byte[] quantizedVector = SegmentLevelQuantizationUtil.quantizeVector(vectorCopy, segmentLevelQuantizationInfo);
            return SpaceType.HAMMING.getKnnVectorSimilarityFunction().compare(quantizedQueryVector, quantizedVector);
        } else {
            // Calculates a similarity score between the two vectors with a specified function. Higher similarity
            // scores correspond to closer vectors.
            return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, vectorCopy);
        }
    }

    /** @return the score for the doc most recently returned by nextDoc() on this thread */
    @Override
    public float score() {
        return currentScore.get();
    }

    /** @return the docID for the doc most recently returned by nextDoc() on this thread */
    public int docID() {
        return docId.get();
    }

    /**
     * Fetch the next doc from either the filterIdsIterator or
     * the raw knnFloatVectorValues.
     */
    protected int getNextDocId() throws IOException {
        if (filterIdsIterator == null) {
            return knnFloatVectorValues.nextDoc();
        }
        int next = filterIdsIterator.nextDoc();
        if (next != DocIdSetIterator.NO_MORE_DOCS) {
            knnFloatVectorValues.advance(next);
        }
        return next;
    }
}

