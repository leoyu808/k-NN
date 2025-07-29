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

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.util.IndexUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.Optional;

import static org.opensearch.knn.index.codec.KNN990Codec.NativeEngineSegmentAttributeParser.INDEX_NAME;

/**
 * This is a Vector format that will be used for Native engines like Faiss and Nmslib for reading and writing vector
 * related data structures.
 */
@Log4j2
public class NativeEngines990KnnVectorsFormat extends KnnVectorsFormat {
    /** The format for storing, reading, merging vectors on disk */
    private static FlatVectorsFormat flatVectorsFormat;
    private static final String FORMAT_NAME = "NativeEngines990KnnVectorsFormat";
    private static int approximateThreshold;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;
    private final Optional<MapperService> mapperService;

    public NativeEngines990KnnVectorsFormat() {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()));
    }

    public NativeEngines990KnnVectorsFormat(int approximateThreshold) {
        this(new Lucene99FlatVectorsFormat(new DefaultFlatVectorScorer()), approximateThreshold);
    }

    public NativeEngines990KnnVectorsFormat(final FlatVectorsFormat flatVectorsFormat) {
        this(flatVectorsFormat, KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
    }

    public NativeEngines990KnnVectorsFormat(final FlatVectorsFormat flatVectorsFormat, int approximateThreshold) {
        this(flatVectorsFormat, approximateThreshold, new NativeIndexBuildStrategyFactory(), Optional.empty());
    }

    public NativeEngines990KnnVectorsFormat(
        final FlatVectorsFormat flatVectorsFormat,
        int approximateThreshold,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        final Optional<MapperService> mapperService
    ) {
        super(FORMAT_NAME);
        NativeEngines990KnnVectorsFormat.flatVectorsFormat = flatVectorsFormat;
        NativeEngines990KnnVectorsFormat.approximateThreshold = approximateThreshold;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
        this.mapperService = mapperService;
    }

    /**
     * Returns a {@link KnnVectorsWriter} to write the vectors to the index.
     *
     * @param state {@link SegmentWriteState}
     */
    @Override
    public KnnVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
        state.segmentInfo.putAttribute(INDEX_NAME, mapperService.get().index().getName());
        IndexOutput output = state.directory.createOutput(IndexFileNames.segmentFileName(state.segmentInfo.name, "", "mem"), state.context);
        CodecUtil.writeIndexHeader(output, "NativeEngines990KnnVectors", 0, state.segmentInfo.getId(), state.segmentSuffix);
        CodecUtil.writeFooter(output);
        output.close();
        return new NativeEngines990KnnVectorsWriter(
            state,
            flatVectorsFormat.fieldsWriter(state),
            approximateThreshold,
            nativeIndexBuildStrategyFactory
        );
    }

    /**
     * Returns a {@link KnnVectorsReader} to read the vectors from the index.
     *
     * @param state {@link SegmentReadState}
     */
    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        return new NativeEngines990KnnVectorsReader(state, flatVectorsFormat.fieldsReader(state));
    }

    /**
     * @param s
     * @return
     */
    @Override
    public int getMaxDimensions(String s) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return "NativeEngines99KnnVectorsFormat(name="
            + this.getClass().getSimpleName()
            + ", flatVectorsFormat="
            + flatVectorsFormat
            + ", approximateThreshold="
            + approximateThreshold
            + ")";
    }
}
