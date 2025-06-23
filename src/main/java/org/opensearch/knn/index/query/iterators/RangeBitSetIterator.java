/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;

public class RangeBitSetIterator extends DocIdSetIterator {
    // can be shared across partitions
    private final BitSet bits;
    private final int length;
    private int doc;
    private long cost;
    public final int minDocId;
    public final int maxDocId;

    public RangeBitSetIterator(BitSet bits, long cost, int minDocId, int maxDocId) {
        if (cost < 0) {
            throw new IllegalArgumentException("cost must be >= 0, got " + cost);
        }
        this.bits = bits;
        this.length = bits.length();
        this.cost = cost;
        this.minDocId = minDocId;
        this.maxDocId = maxDocId;
        doc = minDocId - 1;
    }

    public BitSet getBitSet() {
        return bits;
    }

    @Override
    public int docID() {
        return doc;
    }

    public void setDocId(int docId) {
        this.doc = docId;
    }

    @Override
    public int nextDoc() {
        return advance(doc + 1);
    }

    @Override
    public int advance(int target) {
        if (target >= length || target >= maxDocId) {
            return doc = NO_MORE_DOCS;
        }
        return doc = bits.nextSetBit(target);
    }

    @Override
    public long cost() {
        return cost;
    }
}
