/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.*;

import java.io.IOException;

import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryEntryContext;
import org.opensearch.knn.index.memory.NativeMemoryLoadStrategy;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;

import static org.opensearch.knn.index.util.IndexUtil.getParametersAtLoading;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Log4j2
public class KNN80DocValuesProducer extends DocValuesProducer {
    private final DocValuesProducer delegate;
    private List<String> cacheKeys;

    public KNN80DocValuesProducer(DocValuesProducer delegate, SegmentReadState state) {
        this.delegate = delegate;
        this.cacheKeys = getVectorCacheKeysFromSegmentReaderState(state);
        warmUpIndices(state);
    }

    @Override
    public BinaryDocValues getBinary(FieldInfo field) throws IOException {
        return delegate.getBinary(field);
    }

    @Override
    public NumericDocValues getNumeric(FieldInfo field) throws IOException {
        return delegate.getNumeric(field);
    }

    @Override
    public SortedDocValues getSorted(FieldInfo field) throws IOException {
        return delegate.getSorted(field);
    }

    @Override
    public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
        return delegate.getSortedNumeric(field);
    }

    @Override
    public SortedSetDocValues getSortedSet(FieldInfo field) throws IOException {
        return delegate.getSortedSet(field);
    }

    /**
     * @param fieldInfo
     * @return Returns a DocValuesSkipper for this field. The returned instance need not be thread-safe:
     * it will only be used by a single thread.
     * The return value is undefined if FieldInfo. docValuesSkipIndexType() returns DocValuesSkipIndexType. NONE.
     * @throws IOException
     */
    @Override
    public DocValuesSkipper getSkipper(FieldInfo fieldInfo) throws IOException {
        return delegate.getSkipper(fieldInfo);
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegate.checkIntegrity();
    }

    @Override
    public void close() throws IOException {
        final NativeMemoryCacheManager nativeMemoryCacheManager = NativeMemoryCacheManager.getInstance();
        cacheKeys.forEach(nativeMemoryCacheManager::invalidate);
        delegate.close();
    }

    public final List<String> getCacheKeys() {
        return new ArrayList<>(cacheKeys);
    }

    private static List<String> getVectorCacheKeysFromSegmentReaderState(SegmentReadState segmentReadState) {
        final List<String> cacheKeys = new ArrayList<>();

        for (FieldInfo field : segmentReadState.fieldInfos) {
            // Only segments that contains BinaryDocValues and doesn't have vector values should be considered.
            // By default, we don't create BinaryDocValues for knn field anymore. However, users can set doc_values = true
            // to create binary doc values explicitly like any other field. Hence, we only want to include fields
            // where approximate search is possible only by BinaryDocValues.
            if (field.getDocValuesType() != DocValuesType.BINARY || field.hasVectorValues()) {
                continue;
            }

            final String vectorIndexFileName = KNNCodecUtil.getNativeEngineFileFromFieldInfo(field, segmentReadState.segmentInfo);
            if (vectorIndexFileName == null) {
                continue;
            }
            final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(vectorIndexFileName, segmentReadState.segmentInfo);
            cacheKeys.add(cacheKey);
        }

        return cacheKeys;
    }

    private void warmUpIndices(final SegmentReadState segmentReadState) {
        String indexName = segmentReadState.segmentInfo.getAttribute("index_name");
        String warmupEnabled = segmentReadState.segmentInfo.getAttribute("warmup_enabled");
        if (indexName != null && warmupEnabled.equals("true")) {
            for (final FieldInfo fieldInfo : segmentReadState.fieldInfos) {
                final String vectorIndexFileName = KNNCodecUtil.getNativeEngineFileFromFieldInfo(fieldInfo, segmentReadState.segmentInfo);
                if (vectorIndexFileName == null) {
                    continue;
                }
                final String cacheKey = NativeMemoryCacheKeyHelper.constructCacheKey(vectorIndexFileName, segmentReadState.segmentInfo);
                final NativeMemoryCacheManager cacheManager = NativeMemoryCacheManager.getInstance();
                try {
                    final String spaceTypeName = fieldInfo.attributes().getOrDefault(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
                    final SpaceType spaceType = SpaceType.getSpace(spaceTypeName);
                    final KNNEngine knnEngine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
                    final VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
                    final QuantizationParams quantizationParams = QuantizationService.getInstance()
                        .getQuantizationParams(fieldInfo, segmentReadState.segmentInfo.getVersion());
                    cacheManager.get(
                        new NativeMemoryEntryContext.IndexEntryContext(
                            segmentReadState.segmentInfo,
                            segmentReadState.segmentInfo.dir,
                            cacheKey,
                            NativeMemoryLoadStrategy.IndexLoadStrategy.getInstance(),
                            getParametersAtLoading(spaceType, knnEngine, indexName, vectorDataType, quantizationParams),
                            indexName
                        ),
                        true
                    );
                } catch (ExecutionException e) {

                }
            }
        }
    }
}
