/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.iterators;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.io.IOException;

public interface KNNIterator {
    int nextDoc() throws IOException;

    float score();

    @Getter
    @AllArgsConstructor
    class DocVector {
        private final int docId;
        private final float[] vector;
    }
}
