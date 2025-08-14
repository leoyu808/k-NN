/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import com.google.common.annotations.VisibleForTesting;
import org.apache.lucene.index.SegmentInfo;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class NativeMemoryCacheRegistryManager {
    private static NativeMemoryCacheRegistryManager INSTANCE;

    private Map<String, NativeMemoryCacheRegistry> registries;
    private final ExecutorService executor;

    /**
     * Make sure we just have one instance of cache.
     *
     * @return NativeMemoryCacheManager instance
     */
    public static synchronized NativeMemoryCacheRegistryManager getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new NativeMemoryCacheRegistryManager();
        }
        return INSTANCE;
    }

    @VisibleForTesting
    public static void setInstance(NativeMemoryCacheRegistryManager nativeMemoryCacheRegistryManager) {
        INSTANCE = nativeMemoryCacheRegistryManager;
    }

    NativeMemoryCacheRegistryManager() {
        this.executor = Executors.newSingleThreadExecutor();
        this.registries = new HashMap<>();
    }

    public void openSegmentRegistry(SegmentInfo segmentInfo) throws IOException {
        String segmentKey = NativeMemoryCacheKeyHelper.constructSegmentKey(segmentInfo);
        registries.put(segmentKey, new NativeMemoryCacheRegistry(segmentInfo));
    }

    public void closeSegmentRegistry(SegmentInfo segmentInfo) {
        String segmentKey = NativeMemoryCacheKeyHelper.constructSegmentKey(segmentInfo);
        registries.remove(segmentKey);
    }

    public void addFileSegmentRegistry(SegmentInfo segmentInfo, String fileName) {
        String segmentKey = NativeMemoryCacheKeyHelper.constructSegmentKey(segmentInfo);
        NativeMemoryCacheRegistry registry = registries.get(segmentKey);
        if (registry == null) return;
        registry.addFile(fileName);
    }

    public void deleteFileSegmentRegistry(SegmentInfo segmentInfo, String fileName) {
        String segmentKey = NativeMemoryCacheKeyHelper.constructSegmentKey(segmentInfo);
        NativeMemoryCacheRegistry registry = registries.get(segmentKey);
        if (registry == null) return;
        registry.deleteFile(fileName);
    }

    public boolean containsFileSegmentRegistry(SegmentInfo segmentInfo, String fileName) {
        String segmentKey = NativeMemoryCacheKeyHelper.constructSegmentKey(segmentInfo);
        NativeMemoryCacheRegistry registry = registries.get(segmentKey);
        if (registry == null) return false;
        return registry.containsFile(fileName);
    }
}
