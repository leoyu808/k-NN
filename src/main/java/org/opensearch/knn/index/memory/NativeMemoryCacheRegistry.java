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
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

@Log4j2
public class NativeMemoryCacheRegistry {
    private final SegmentInfo segmentInfo;
    private final Directory directory;
    private final String registryFileName;
    private final String tempRegistryFileName;
    private final String backupRegistryFileName;
    private final Set<String> inMemory;
    private final boolean registryFileExists;
    private final Lock lock = new ReentrantLock();

    public NativeMemoryCacheRegistry(SegmentInfo segmentInfo) throws IOException {
        this.segmentInfo = segmentInfo;
        directory = segmentInfo.dir;
        String memExtension = segmentInfo.getUseCompoundFile()
            ? KNNConstants.NATIVE_ENGINE_MEMORY_STATE_SUFFIX + KNNConstants.COMPOUND_EXTENSION
            : KNNConstants.NATIVE_ENGINE_MEMORY_STATE_SUFFIX;
        registryFileName = IndexFileNames.segmentFileName(segmentInfo.name, "", memExtension);
        tempRegistryFileName = IndexFileNames.segmentFileName(segmentInfo.name, "", memExtension + KNNConstants.TEMP_FILE_SUFFIX);
        backupRegistryFileName = IndexFileNames.segmentFileName(segmentInfo.name, "", memExtension + KNNConstants.BACKUP_FILE_SUFFIX);
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

    public void addFile(String fileName) {
        lock.lock();
        try {
            if (!inMemory.contains(fileName) && registryFileExists) {
                inMemory.add(fileName);
                writeState();
            }
        } finally {
            lock.unlock();
        }
    }

    public void deleteFile(String fileName) {
        lock.lock();
        try {
            if (inMemory.contains(fileName) && registryFileExists) {
                inMemory.remove(fileName);
                writeState();
            }
        } finally {
            lock.unlock();
        }
    }

    public boolean containsFile(String fileName) {
        return inMemory.contains(fileName);
    }

    private void writeState() {
        try {
            try (
                IndexInput in = directory.openInput(registryFileName, IOContext.READONCE);
                IndexOutput out = directory.createOutput(tempRegistryFileName, IOContext.DEFAULT)
            ) {
                CodecUtil.verifyAndCopyIndexHeader(in, out, segmentInfo.getId());
                for (String file : inMemory) {
                    out.writeString(file);
                }
                CodecUtil.writeFooter(out);
            }
            directory.rename(registryFileName, backupRegistryFileName);
            directory.rename(tempRegistryFileName, registryFileName);
            directory.deleteFile(backupRegistryFileName);
        } catch (IOException e) {
            log.error("Failed to write graph files to disk: {}", registryFileName);
        }
    }
}
