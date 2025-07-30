/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.codec.util.NativeMemoryCacheKeyHelper;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class NativeMemoryCacheRegistry {
    private final Directory directory;
    private final String registryFileName;
    private final Set<String> inMemory;

    public NativeMemoryCacheRegistry(SegmentInfo segmentInfo) throws IOException {
        directory = segmentInfo.dir;
        registryFileName = IndexFileNames.segmentFileName(segmentInfo.name, "", "mem");
        inMemory = new HashSet<>();

        try (IndexInput in = directory.openInput(registryFileName, IOContext.READONCE)) {
            CodecUtil.readIndexHeader(in);
            long footerStart = in.length() - CodecUtil.footerLength();
            while (in.getFilePointer() < footerStart) {
                String s = in.readString();
                inMemory.add(s);
            }
        }
    }

    public void addFile(String fileName) throws IOException {
        if (!inMemory.contains(fileName)) {
            inMemory.add(fileName);
            persistToDisk();
        }
    }

    public void deleteFile(String fileName) throws IOException {
        if (inMemory.contains(fileName)) {
            inMemory.remove(fileName);
            persistToDisk();
        }
    }

    public boolean containsFile(String fileName) {
        return inMemory.contains(fileName);
    }

    private void persistToDisk() throws IOException {
        byte[] oldHeaderBytes;
        try (IndexInput in = directory.openInput(registryFileName, IOContext.READONCE)) {
            oldHeaderBytes = CodecUtil.readIndexHeader(in);
            directory.deleteFile(registryFileName);
        }
        try (IndexOutput out = directory.createOutput(registryFileName, IOContext.DEFAULT)) {
            out.writeBytes(oldHeaderBytes, oldHeaderBytes.length);
            for (String name : inMemory) {
                out.writeString(name);
            }
            CodecUtil.writeFooter(out);
        }
    }
}