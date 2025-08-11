/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

@Log4j2
public class NativeMemoryCacheRegistry {
    private final Directory directory;
    private final String registryFileName;
    private final Set<String> inMemory;
    private final boolean registryFileExists;

    public NativeMemoryCacheRegistry(SegmentInfo segmentInfo) throws IOException {
        directory = segmentInfo.dir;
        String memExtension = segmentInfo.getUseCompoundFile()
            ? KNNConstants.CACHE_MARKER + KNNConstants.COMPOUND_EXTENSION
            : KNNConstants.CACHE_MARKER;
        registryFileName = IndexFileNames.segmentFileName(segmentInfo.name, "", memExtension);
        inMemory = new HashSet<>();
        registryFileExists = Arrays.asList(directory.listAll()).contains(registryFileName);
        if (!registryFileExists) {
            return;
        }
        try (IndexInput in = directory.openInput(registryFileName, IOContext.READONCE)) {
            CodecUtil.readIndexHeader(in);
            long footerStart = in.length() - CodecUtil.footerLength();
            while (in.getFilePointer() < footerStart) {
                String fileName = in.readString();
                inMemory.add(fileName);
            }
        }
    }

    public void addFile(String fileName) throws IOException {
        if (!inMemory.contains(fileName) && registryFileExists) {
            inMemory.add(fileName);
            persistToDisk();
        }
    }

    public void deleteFile(String fileName) throws IOException {
        if (inMemory.contains(fileName) && registryFileExists) {
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
            for (String fileName : inMemory) {
                out.writeString(fileName);
            }
            CodecUtil.writeFooter(out);
        }
    }
}
