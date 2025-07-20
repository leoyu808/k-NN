/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.base.Predicates;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NonNull;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.*;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.iterators.*;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNBinaryVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNByteVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNFloatVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.function.Predicate;

import static org.opensearch.knn.common.KNNConstants.SEARCH_THREAD_POOL;

@Log4j2
@AllArgsConstructor
public class ExactSearcher {

    private final ModelDao modelDao;
    private static ForkJoinPool pool;

    public static void initialize(ThreadPool threadPool) {
        ExactSearcher.pool = (ForkJoinPool) threadPool.executor(SEARCH_THREAD_POOL);
    }

    /**
     * Execute an exact search on a subset of documents of a leaf
     *
     * @param leafReaderContext {@link LeafReaderContext}
     * @param context {@link ExactSearcherContext}
     * @return TopDocs containing the results of the search
     * @throws IOException exception during execution of exact search
     */
    public TopDocs searchLeaf(final LeafReaderContext leafReaderContext, final ExactSearcherContext context) throws IOException {
        // because of any reason if we are not able to get KNNIterator, return empty top docss
        if (context.getRadius() != null) {
            return doRadialSearch(leafReaderContext, context);
        }
        if (context.getFilterBitSet() != null && context.numberOfMatchedDocs <= context.getK()) {
            return scoreAllDocs(leafReaderContext, context);
        }
        return partitionLeaf(leafReaderContext, context, context.getK(), Predicates.alwaysTrue());
    }

    /**
     * Perform radial search by comparing scores with min score. Currently, FAISS from native engine supports radial search.
     * Hence, we assume that Radius from knnQuery is always distance, and we convert it to score since we do exact search uses scores
     * to filter out the documents that does not have given min score.
     * @param leafReaderContext {@link LeafReaderContext}
     * @param context {@link ExactSearcherContext}
     * @return TopDocs containing the results of the search
     * @throws IOException exception raised by iterator during traversal
     */
    private TopDocs doRadialSearch(LeafReaderContext leafReaderContext, ExactSearcherContext context) throws IOException {
        // Ensure `isMemoryOptimizedSearchEnabled` is set. This is necessary to determine whether distance to score conversion is required.
        assert (context.isMemoryOptimizedSearchEnabled != null);

        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, context.getField());
        if (fieldInfo == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        final KNNEngine engine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
        if (KNNEngine.FAISS != engine) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Engine [%s] does not support radial search", engine));
        }
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        // We need to use the given radius when memory optimized search is enabled. Since it relies on Lucene's scoring framework, the given
        // max distance is already converted min score then saved in `radius`. Thus, we don't need a score translation which does not make
        // sense as it is treating min score as a max distance otherwise.
        final float minScore = context.isMemoryOptimizedSearchEnabled
            ? context.getRadius()
            : spaceType.scoreTranslation(context.getRadius());
        return filterDocsByMinScore(leafReaderContext, context, minScore);
    }

    private TopDocs scoreAllDocs(LeafReaderContext leafReaderContext, ExactSearcherContext context) throws IOException {
        BitSet filterBitSet = context.getFilterBitSet();
        BitSetIterator matchedDocs = filterBitSet != null ? new BitSetIterator(filterBitSet, context.getNumberOfMatchedDocs()) : null;
        KNNIterator iterator = getKNNIterator(leafReaderContext, context, matchedDocs);
        if (iterator == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        final List<ScoreDoc> scoreDocList = new ArrayList<>();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            scoreDocList.add(new ScoreDoc(docId, iterator.score()));
        }
        scoreDocList.sort(Comparator.comparing(scoreDoc -> scoreDoc.score, Comparator.reverseOrder()));
        return new TopDocs(new TotalHits(scoreDocList.size(), TotalHits.Relation.EQUAL_TO), scoreDocList.toArray(ScoreDoc[]::new));
    }

    private TopDocs partitionLeaf(
        LeafReaderContext leafReaderContext,
        ExactSearcherContext context,
        int limit,
        @NonNull Predicate<Float> filterScore
    ) throws IOException {
        try {
            return ExactSearcher.pool.invoke(
                new ExactSearchTask(leafReaderContext, context, limit, filterScore, 0, leafReaderContext.reader().maxDoc())
            );
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
    }

    private TopDocs searchTopCandidates(
        LeafReaderContext leafReaderContext,
        ExactSearcherContext context,
        int limit,
        @NonNull Predicate<Float> filterScore,
        int minDocId,
        int maxDocId
    ) throws IOException {
        log.debug("Working Thread: {}", Thread.currentThread().getName());
        BitSet filterBitSet = context.getFilterBitSet();
        DocIdSetIterator matchedDocs = filterBitSet != null
            ? new RangeDocIdSetIterator(new BitSetIterator(filterBitSet, context.getNumberOfMatchedDocs()), minDocId, maxDocId)
            : DocIdSetIterator.range(minDocId, maxDocId);
        KNNIterator iterator = getKNNIterator(leafReaderContext, context, matchedDocs);
        if (iterator == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        final HitQueue queue = new HitQueue(limit, true);
        ScoreDoc topDoc = queue.top();
        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            final float currentScore = iterator.score();
            if (filterScore.test(currentScore) && currentScore > topDoc.score) {
                topDoc.score = currentScore;
                topDoc.doc = docId;
                // As the HitQueue is min heap, updating top will bring the doc with -INF score or worst score we
                // have seen till now on top.
                topDoc = queue.updateTop();
            }
        }

        // If scores are negative we will remove them.
        // This is done, because there can be negative values in the Heap as we init the heap with Score as -INF.
        // If filterIds < k, some values in heap can have a negative score.
        while (queue.size() > 0 && queue.top().score < 0) {
            queue.pop();
        }

        ScoreDoc[] topScoreDocs = new ScoreDoc[queue.size()];
        for (int j = topScoreDocs.length - 1; j >= 0; j--) {
            topScoreDocs[j] = queue.pop();
        }

        TotalHits totalHits = new TotalHits(topScoreDocs.length, TotalHits.Relation.EQUAL_TO);
        return new TopDocs(totalHits, topScoreDocs);
    }

    private TopDocs filterDocsByMinScore(LeafReaderContext leafReaderContext, ExactSearcherContext context, float minScore)
        throws IOException {
        int maxResultWindow = context.getMaxResultWindow();
        Predicate<Float> scoreGreaterThanOrEqualToMinScore = score -> score >= minScore;
        return partitionLeaf(leafReaderContext, context, maxResultWindow, scoreGreaterThanOrEqualToMinScore);
    }

    private KNNIterator getKNNIterator(
        LeafReaderContext leafReaderContext,
        ExactSearcherContext exactSearcherContext,
        DocIdSetIterator matchedDocs
    ) throws IOException {
        final SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, exactSearcherContext.getField());
        if (fieldInfo == null) {
            log.debug(
                "[KNN] Cannot get KNNIterator as Field info not found for {}:{}",
                exactSearcherContext.getField(),
                reader.getSegmentName()
            );
            return null;
        }
        final VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
        final SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        boolean isNestedRequired = exactSearcherContext.getParentsFilter() != null;

        if (VectorDataType.BINARY == vectorDataType) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedBinaryVectorIdsKNNIterator(
                    matchedDocs,
                    exactSearcherContext.getByteQueryVector(),
                    (KNNBinaryVectorValues) vectorValues,
                    spaceType,
                    exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new BinaryVectorIdsKNNIterator(
                matchedDocs,
                exactSearcherContext.getByteQueryVector(),
                (KNNBinaryVectorValues) vectorValues,
                spaceType
            );
        }

        if (VectorDataType.BYTE == vectorDataType) {
            final KNNVectorValues<byte[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
            if (isNestedRequired) {
                return new NestedByteVectorIdsKNNIterator(
                    matchedDocs,
                    exactSearcherContext.getFloatQueryVector(),
                    (KNNByteVectorValues) vectorValues,
                    spaceType,
                    exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext)
                );
            }
            return new ByteVectorIdsKNNIterator(
                matchedDocs,
                exactSearcherContext.getFloatQueryVector(),
                (KNNByteVectorValues) vectorValues,
                spaceType
            );
        }
        final byte[] quantizedQueryVector;
        final SegmentLevelQuantizationInfo segmentLevelQuantizationInfo;
        if (exactSearcherContext.isUseQuantizedVectorsForSearch()) {
            // Build Segment Level Quantization info.
            segmentLevelQuantizationInfo = SegmentLevelQuantizationInfo.build(reader, fieldInfo, exactSearcherContext.getField());
            // Quantize the Query Vector Once.
            quantizedQueryVector = SegmentLevelQuantizationUtil.quantizeVector(
                exactSearcherContext.getFloatQueryVector(),
                segmentLevelQuantizationInfo
            );
        } else {
            segmentLevelQuantizationInfo = null;
            quantizedQueryVector = null;
        }

        final KNNVectorValues<float[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        if (isNestedRequired) {
            return new NestedVectorIdsKNNIterator(
                matchedDocs,
                exactSearcherContext.getFloatQueryVector(),
                (KNNFloatVectorValues) vectorValues,
                spaceType,
                exactSearcherContext.getParentsFilter().getBitSet(leafReaderContext),
                quantizedQueryVector,
                segmentLevelQuantizationInfo
            );
        }
        return new VectorIdsKNNIterator(
            matchedDocs,
            exactSearcherContext.getFloatQueryVector(),
            (KNNFloatVectorValues) vectorValues,
            spaceType,
            quantizedQueryVector,
            segmentLevelQuantizationInfo
        );
    }

    /**
     * Stores the context that is used to do the exact search. This class will help in reducing the explosion of attributes
     * for doing exact search.
     */
    @Value
    @Builder
    public static class ExactSearcherContext {
        /**
         * controls whether we should use Quantized vectors during exact search or not. This is useful because when we do
         * re-scoring we need to re-score using full precision vectors and not quantized vectors.
         */
        boolean useQuantizedVectorsForSearch;
        int k;
        Float radius;
        BitSet filterBitSet;
        long numberOfMatchedDocs;
        /**
         * whether the matchedDocs contains parent ids or child ids. This is relevant in the case of
         * filtered nested search where the matchedDocs contain the parent ids and {@link NestedVectorIdsKNNIterator}
         * needs to be used.
         */
        BitSetProducer parentsFilter;
        float[] floatQueryVector;
        byte[] byteQueryVector;
        String field;
        Integer maxResultWindow;
        VectorSimilarityFunction similarityFunction;
        Boolean isMemoryOptimizedSearchEnabled;
    }

    public class ExactSearchTask extends RecursiveTask<TopDocs> {
        final LeafReaderContext leafReaderContext;
        final ExactSearcherContext context;
        final int limit;
        final Predicate<Float> filterScore;
        final int THRESHOLD = 100_000;
        final int minDocId;
        final int maxDocId;

        public ExactSearchTask(
            LeafReaderContext leafReaderContext,
            ExactSearcherContext context,
            int limit,
            @NonNull Predicate<Float> filterScore,
            int minDocId,
            int maxDocId
        ) {
            this.leafReaderContext = leafReaderContext;
            this.context = context;
            this.limit = limit;
            this.filterScore = filterScore;
            this.minDocId = minDocId;
            this.maxDocId = maxDocId;
        }

        @Override
        protected TopDocs compute() {
            if (maxDocId - minDocId < THRESHOLD) {
                try {
                    return searchTopCandidates(leafReaderContext, context, limit, filterScore, minDocId, maxDocId);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            } else {
                int mid = minDocId + (maxDocId - minDocId) / 2;
                ExactSearchTask left = new ExactSearchTask(leafReaderContext, context, limit, filterScore, minDocId, mid);
                ExactSearchTask right = new ExactSearchTask(leafReaderContext, context, limit, filterScore, mid, maxDocId);
                left.fork();
                right.fork();
                return TopDocs.merge(context.getK(), new TopDocs[] { left.join(), right.join() });
            }
        }
    }
}
