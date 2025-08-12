/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.index.SegmentInfo;
import org.opensearch.index.mapper.MapperService;

public class NativeEngineSegmentAttributeParser {

    static final String INDEX_NAME = "index_name";

    /**
     * From segmentInfo, parse index name
     *
     * @param segmentInfo {@link SegmentInfo}
     * @return String of the name of the index the segment is a member of
     */
    public static String parseIndexName(SegmentInfo segmentInfo) {
        if (segmentInfo == null) {
            throw new IllegalArgumentException("SegmentInfo cannot be null");
        }
        return segmentInfo.getAttribute(INDEX_NAME);
    }

    /**
     * Adds {@link SegmentInfo} attribute for warmup
     *
     * @param segmentInfo {@link SegmentInfo}
     */
    public static void writeIndexName(MapperService mapperService, SegmentInfo segmentInfo) {
        if (segmentInfo == null) {
            throw new IllegalArgumentException("SegmentInfo cannot be null");
        }
        if (mapperService == null) {
            throw new IllegalArgumentException("MapperService cannot be null");
        }
        String indexName = mapperService.index().getName();
        segmentInfo.putAttribute(INDEX_NAME, indexName);
    }
}
