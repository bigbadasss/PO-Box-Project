'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Upload, Camera, Eye, EyeOff, Play, Square } from 'lucide-react';
import Papa from 'papaparse';
import { createWorker } from 'tesseract.js';

interface CSVRow {
  name: string;
  address: string;
  email: string;
  [key: string]: any;
}

interface MatchResult {
  row: CSVRow;
  matchedFields: string[];
  confidence: number;
  similarity: number;
  ocrText: string;
  matchedSegment: string;
  fieldScores?: {[key: string]: number};
  keywordMatches?: number;
}

interface CSVFileInfo {
  name: string;
  uploadTime: string;
  size: number;
  data: CSVRow[];
}

// Enhanced fuzzy matching utilities
const calculateLevenshteinDistance = (str1: string, str2: string): number => {
  const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null));
  
  for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
  for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;
  
  for (let j = 1; j <= str2.length; j++) {
    for (let i = 1; i <= str1.length; i++) {
      if (str1[i - 1] === str2[j - 1]) {
        matrix[j][i] = matrix[j - 1][i - 1];
      } else {
        matrix[j][i] = Math.min(
          matrix[j - 1][i] + 1,
          matrix[j][i - 1] + 1,
          matrix[j - 1][i - 1] + 1
        );
      }
    }
  }
  
  return matrix[str2.length][str1.length];
};

// 改进的文本标准化函数
const normalizeString = (str: string): string => {
  return str
    .toLowerCase()
    .trim()
    .replace(/\s+/g, '') // 移除所有空格
    .replace(/[^\w\u4e00-\u9fff]/g, '') // 只保留字母、数字、中文字符
    .replace(/[0-9]/g, ''); // 移除数字，因为OCR经常识别错误
};

// 提取关键词（中文姓名、地址关键词等）
const extractKeywords = (text: string): string[] => {
  const keywords: string[] = [];
  
  // 提取中文姓名（2-4个字符）
  const nameMatches = text.match(/[\u4e00-\u9fff]{2,4}/g) || [];
  keywords.push(...nameMatches);
  
  // 提取地址关键词
  const addressKeywords = ['区', '市', '省', '路', '街', '号', '楼', '室', '单元'];
  addressKeywords.forEach(keyword => {
    if (text.includes(keyword)) {
      const context = text.match(new RegExp(`[\\u4e00-\\u9fff]*${keyword}[\\u4e00-\\u9fff]*`, 'g')) || [];
      keywords.push(...context);
    }
  });
  
  return [...new Set(keywords)]; // 去重
};

// 计算字符串相似度的多种算法
const calculateSimilarity = (str1: string, str2: string): number => {
  const cleanStr1 = normalizeString(str1);
  const cleanStr2 = normalizeString(str2);
  
  if (!cleanStr1 || !cleanStr2) return 0;
  if (cleanStr1 === cleanStr2) return 1;
  
  // 完全包含关系
  if (cleanStr1.includes(cleanStr2) || cleanStr2.includes(cleanStr1)) {
    const longer = cleanStr1.length > cleanStr2.length ? cleanStr1 : cleanStr2;
    const shorter = cleanStr1.length > cleanStr2.length ? cleanStr2 : cleanStr1;
    return 0.9 + (shorter.length / longer.length) * 0.1;
  }
  
  // 前缀匹配（对于姓名特别有用）
  if (cleanStr1.length >= 2 && cleanStr2.length >= 2) {
    const prefix1 = cleanStr1.substring(0, 2);
    const prefix2 = cleanStr2.substring(0, 2);
    if (prefix1 === prefix2) {
      const suffixSimilarity = calculateLevenshteinDistance(
        cleanStr1.substring(2), 
        cleanStr2.substring(2)
      ) / Math.max(cleanStr1.length - 2, cleanStr2.length - 2, 1);
      return 0.8 + (1 - suffixSimilarity) * 0.2;
    }
  }
  
  // Levenshtein距离
  const distance = calculateLevenshteinDistance(cleanStr1, cleanStr2);
  const maxLength = Math.max(cleanStr1.length, cleanStr2.length);
  const levenshteinSimilarity = maxLength === 0 ? 0 : 1 - distance / maxLength;
  
  // Jaccard相似度（字符级别）
  const set1 = new Set(cleanStr1.split(''));
  const set2 = new Set(cleanStr2.split(''));
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);
  const jaccardSimilarity = union.size === 0 ? 0 : intersection.size / union.size;
  
  // 关键词匹配
  const keywords1 = extractKeywords(str1);
  const keywords2 = extractKeywords(str2);
  const keywordMatches = keywords1.filter(k1 => 
    keywords2.some(k2 => calculateLevenshteinDistance(k1, k2) <= 1)
  ).length;
  const keywordSimilarity = keywords1.length > 0 && keywords2.length > 0 
    ? keywordMatches / Math.max(keywords1.length, keywords2.length) 
    : 0;
  
  // 综合相似度计算
  return (
    levenshteinSimilarity * 0.5 + 
    jaccardSimilarity * 0.2 + 
    keywordSimilarity * 0.3
  );
};

const findBestSubstring = (text: string, target: string): string => {
  const normalizedText = normalizeString(text);
  const normalizedTarget = normalizeString(target);
  
  if (normalizedText.includes(normalizedTarget)) {
    const startIndex = normalizedText.indexOf(normalizedTarget);
    let originalStartIndex = 0;
    let normalizedIndex = 0;
    for (let i = 0; i < text.length; i++) {
      const normalizedChar = normalizeString(text[i]);
      if (normalizedIndex === startIndex) {
        originalStartIndex = i;
        break;
      }
      normalizedIndex += normalizedChar.length;
    }
    return text.substring(originalStartIndex, originalStartIndex + normalizedTarget.length);
  }
  return '';
};

const findBestMatch = (ocrText: string, csvData: CSVRow[]): MatchResult | null => {
  if (!ocrText.trim() || csvData.length === 0) return null;

  let bestMatch: MatchResult | null = null;
  let bestScore = 0;
  const allMatches: Array<{row: CSVRow, score: number, details: any}> = [];

  // 预处理OCR文本
  const ocrKeywords = extractKeywords(ocrText);
  const normalizedOcr = normalizeString(ocrText);
  
  console.log('OCR预处理:', {
    original: ocrText,
    normalized: normalizedOcr,
    keywords: ocrKeywords
  });

  for (const row of csvData) {
    const matchedFields: string[] = [];
    let totalSimilarity = 0;
    let fieldCount = 0;
    let matchedSegment = '';
    let fieldScores: {[key: string]: number} = {};

    // 检查每个字段
    for (const [field, value] of Object.entries(row)) {
      if (typeof value === 'string' && value.trim()) {
        const similarity = calculateSimilarity(ocrText, value);
        fieldScores[field] = similarity;
        
        // 降低匹配阈值，提高召回率
        if (similarity > 0.2) {
          matchedFields.push(field);
          totalSimilarity += similarity;
          fieldCount++;
          
          // 对于地址字段，提取匹配的文本片段
          if (field === 'address' && similarity > 0.4) {
            matchedSegment = findBestSubstring(ocrText, value);
          }
        }
      }
    }

    if (fieldCount > 0) {
      const averageSimilarity = totalSimilarity / fieldCount;
      
      // 改进的置信度计算
      let confidence = averageSimilarity * 0.6; // 基础相似度权重
      confidence += (fieldCount / Object.keys(row).length) * 0.2; // 匹配字段数量权重
      
      // 关键词匹配权重
      const rowKeywords = extractKeywords(Object.values(row).join(' '));
      const keywordMatches = ocrKeywords.filter(k1 => 
        rowKeywords.some(k2 => calculateLevenshteinDistance(k1, k2) <= 1)
      ).length;
      confidence += (keywordMatches / Math.max(ocrKeywords.length, rowKeywords.length, 1)) * 0.2;
      
      allMatches.push({
        row,
        score: confidence,
        details: {
          fieldScores,
          matchedFields,
          averageSimilarity,
          keywordMatches
        }
      });
      
      if (confidence > bestScore) {
        bestScore = confidence;
        bestMatch = {
          row,
          matchedFields,
          confidence,
          similarity: averageSimilarity,
          ocrText,
          matchedSegment: matchedSegment || findBestSubstring(ocrText, row.address || row.name || ''),
          fieldScores,
          keywordMatches
        };
      }
    }
  }

  // 输出调试信息
  if (allMatches.length > 0) {
    console.log('匹配结果排序:', allMatches
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map(m => ({
        name: m.row.name,
        score: m.score.toFixed(3),
        details: m.details
      }))
    );
  }

  return bestMatch;
};

// OCR文本预处理函数
const preprocessOCRText = (text: string): string => {
  let processed = text;
  
  // 常见OCR错误修正
  const corrections: {[key: string]: string} = {
    '0': 'O', // 数字0误识别为字母O
    '1': 'l', // 数字1误识别为字母l
    '5': 'S', // 数字5误识别为字母S
    '8': 'B', // 数字8误识别为字母B
    'l': '1', // 字母l误识别为数字1
    'O': '0', // 字母O误识别为数字0
    'S': '5', // 字母S误识别为数字5
    'B': '8', // 字母B误识别为数字8
  };
  
  // 应用修正
  Object.entries(corrections).forEach(([wrong, correct]) => {
    processed = processed.replace(new RegExp(wrong, 'g'), correct);
  });
  
  // 移除多余的标点符号和空格
  processed = processed.replace(/[^\w\u4e00-\u9fff\s]/g, ' ').replace(/\s+/g, ' ').trim();
  
  return processed;
};

export default function PoBoxMatchPage() {
  const [csvData, setCsvData] = useState<CSVRow[]>([]);
  const [csvFileInfo, setCsvFileInfo] = useState<CSVFileInfo | null>(null);
  const [matchResults, setMatchResults] = useState<MatchResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isRealTimeActive, setIsRealTimeActive] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [worker, setWorker] = useState<any>(null);
  const [debugInfo, setDebugInfo] = useState<string>('');
  const [lastOcrText, setLastOcrText] = useState<string>('');
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const realTimeIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // 加载缓存的CSV文件
  useEffect(() => {
    const cachedFileInfo = localStorage.getItem('csvFileInfo');
    if (cachedFileInfo) {
      try {
        const parsed = JSON.parse(cachedFileInfo);
        setCsvFileInfo(parsed);
        setCsvData(parsed.data);
      } catch (error) {
        console.error('Failed to parse cached CSV file info:', error);
      }
    }
  }, []);

  // 初始化Tesseract worker
  useEffect(() => {
    const initWorker = async () => {
      try {
        console.log('Initializing Tesseract worker...');
        const newWorker = await createWorker('chi_sim+eng');
        console.log('Tesseract worker initialized successfully');
        setWorker(newWorker);
      } catch (error) {
        console.error('Failed to initialize Tesseract worker:', error);
      }
    };
    initWorker();

    return () => {
      if (worker) {
        worker.terminate();
      }
    };
  }, []);

  // 空格键热键
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !isProcessing) {
        event.preventDefault();
        captureAndProcessOCR(false);
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [isProcessing]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    Papa.parse(file, {
      complete: (results) => {
        const data = results.data as CSVRow[];
        setCsvData(data);
        
        const fileInfo: CSVFileInfo = {
          name: file.name,
          uploadTime: new Date().toLocaleString('zh-CN'),
          size: file.size,
          data: data
        };
        
        setCsvFileInfo(fileInfo);
        localStorage.setItem('csvFileInfo', JSON.stringify(fileInfo));
      },
      header: true,
      skipEmptyLines: true
    });
  };

  const clearCache = () => {
    localStorage.removeItem('csvFileInfo');
    setCsvFileInfo(null);
    setCsvData([]);
    setMatchResults([]);
  };

  const startCamera = async () => {
    try {
      console.log('Starting camera...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // 等待视频加载完成
        videoRef.current.onloadedmetadata = () => {
          console.log('Video loaded, dimensions:', {
            width: videoRef.current?.videoWidth,
            height: videoRef.current?.videoHeight
          });
        };
      }
      console.log('Camera started successfully');
    } catch (error) {
      console.error('Failed to start camera:', error);
      alert('无法启动摄像头，请检查摄像头权限');
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    if (realTimeIntervalRef.current) {
      clearInterval(realTimeIntervalRef.current);
      realTimeIntervalRef.current = null;
    }
    setIsRealTimeActive(false);
  };

  const captureAndProcessOCR = async (isRealTime: boolean = false) => {
    if (!videoRef.current || !canvasRef.current || !worker || isProcessing) {
      console.log('OCR prerequisites not met:', {
        hasVideo: !!videoRef.current,
        hasCanvas: !!canvasRef.current,
        hasWorker: !!worker,
        isProcessing
      });
      return;
    }

    setIsProcessing(true);
    try {
      console.log('Starting OCR processing...');
      
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      if (!context) {
        console.error('Failed to get canvas context');
        return;
      }

      // 确保视频有有效的尺寸
      if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
        console.error('Video dimensions are zero');
        return;
      }

      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      context.drawImage(videoRef.current, 0, 0);

      console.log('Canvas prepared, starting OCR recognition...');
      const { data: { text } } = await worker.recognize(canvas);
      let ocrText = text.trim();
      
      // OCR文本预处理
      ocrText = preprocessOCRText(ocrText);
      
      console.log('OCR result (original):', text.trim());
      console.log('OCR result (processed):', ocrText);
      setLastOcrText(ocrText);
      setDebugInfo(`OCR识别完成: ${ocrText ? ocrText.length : 0} 个字符`);

      if (ocrText && csvData.length > 0) {
        console.log('Processing match with CSV data...');
        const matchResult = findBestMatch(ocrText, csvData);
        if (matchResult) {
          console.log('Match found:', matchResult);
          setMatchResults(prev => [matchResult, ...prev.slice(0, 4)]);
          setDebugInfo(`匹配成功! 置信度: ${(matchResult.confidence * 100).toFixed(1)}%`);
        } else {
          console.log('No match found for OCR text:', ocrText);
          setDebugInfo(`未找到匹配项，OCR文本: ${ocrText}`);
        }
      } else {
        console.log('No OCR text or CSV data available');
        setDebugInfo(ocrText ? '请先上传CSV文件' : 'OCR未识别到文字');
      }
    } catch (error) {
      console.error('OCR processing failed:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const startRealTimeOCR = () => {
    if (!isRealTimeActive) {
      setIsRealTimeActive(true);
      realTimeIntervalRef.current = setInterval(() => {
        captureAndProcessOCR(true);
      }, 2000);
    }
  };

  const stopRealTimeOCR = () => {
    if (realTimeIntervalRef.current) {
      clearInterval(realTimeIntervalRef.current);
      realTimeIntervalRef.current = null;
    }
    setIsRealTimeActive(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">PoBox Match</h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* 左侧：上传和摄像头模块 */}
          <div className="space-y-8">
            {/* CSV文件上传模块 */}
            <Card className="max-w-md lg:mx-0">
              <CardContent className="p-4">
                <p className="text-sm text-gray-600 mb-3 leading-tight font-sans">
                  上传邮箱数据匹配 PO BOX
                </p>
                
                {!csvFileInfo ? (
                  <div className="text-center py-4">
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                      className="hidden"
                      id="csv-upload"
                    />
                    <label htmlFor="csv-upload">
                      <Button asChild className="w-full">
                        <span>
                          <Upload className="w-4 h-4 mr-2" />
                          选择CSV文件
                        </span>
                      </Button>
                    </label>
                  </div>
                ) : (
                  <div>
                    <div className="flex items-center justify-center gap-4 mb-2">
                      <span className="text-sm font-normal font-sans leading-tight">
                        {csvFileInfo.name} {csvFileInfo.uploadTime} {csvFileInfo.data.length} 条记录已加载
                      </span>
                    </div>
                    <div className="flex gap-2">
                      <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileUpload}
                        className="hidden"
                        id="csv-upload-again"
                      />
                      <label htmlFor="csv-upload-again">
                        <Button size="sm" variant="outline" className="text-sm font-sans">
                          <Upload className="w-3 h-3 mr-1" />
                          选择文件
                        </Button>
                      </label>
                      <Button size="sm" variant="outline" onClick={clearCache} className="text-sm font-sans">
                        清除缓存
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 摄像头扫描模块 */}
            <Card className="max-w-md lg:mx-0">
              <CardContent className="p-4">
                <div className="space-y-4">
                  <div className="relative">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-48 bg-gray-200 rounded-lg"
                    />
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      onClick={() => captureAndProcessOCR(false)}
                      disabled={isProcessing || !cameraStream}
                      className="text-xs px-2 py-1 h-8"
                    >
                      <Camera className="w-3 h-3 mr-1" />
                      手动识别
                    </Button>
                    <Button
                      size="sm"
                      variant={isRealTimeActive ? "destructive" : "outline"}
                      onClick={isRealTimeActive ? stopRealTimeOCR : startRealTimeOCR}
                      disabled={!cameraStream}
                      className="text-xs px-2 py-1 h-8"
                    >
                      {isRealTimeActive ? <Square className="w-3 h-3 mr-1" /> : <Play className="w-3 h-3 mr-1" />}
                      {isRealTimeActive ? "停止识别" : "实时识别"}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={cameraStream ? stopCamera : startCamera}
                      className="text-xs px-2 py-1 h-8"
                    >
                      {cameraStream ? "关闭" : "开启"}
                    </Button>
                  </div>
                  
                  {/* 调试信息 */}
                  {debugInfo && (
                    <div className="text-xs text-gray-600 bg-gray-100 p-2 rounded">
                      <div className="font-medium">调试信息:</div>
                      <div>{debugInfo}</div>
                      {lastOcrText && (
                        <div className="mt-1">
                          <div className="font-medium">OCR文本:</div>
                          <div className="bg-white p-1 rounded text-xs">{lastOcrText}</div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 右侧：匹配结果模块 */}
          <div>
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="flex justify-between items-center">
                  <div className="flex items-center">
                    <Eye className="w-5 h-5 mr-2" />
                    匹配结果
                  </div>
                  {matchResults.length > 0 && (
                    <div className="text-2xl font-bold text-red-600">
                      PO Box: {matchResults[0].row.email || matchResults[0].row.address || 'N/A'}
                    </div>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {matchResults.length === 0 ? (
                  <div className="text-center text-gray-500 py-8">
                    请先上传CSV文件并进行OCR识别
                  </div>
                ) : (
                  <div className="space-y-4">
                    {matchResults.map((match, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-4 bg-white">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-900">
                            匹配 #{index + 1}
                          </span>
                        </div>
                        
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium text-gray-700">置信度: </span>
                            <span className="text-sm font-bold text-green-600">
                              {(match.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          
                          <div>
                            <span className="text-sm font-medium text-gray-700">匹配字段: </span>
                            <span className="text-sm text-gray-600">
                              {match.matchedFields.join(', ')}
                            </span>
                          </div>
                          
                          {match.fieldScores && (
                            <div>
                              <span className="text-sm font-medium text-gray-700">字段得分: </span>
                              <div className="text-xs text-gray-500 mt-1">
                                {Object.entries(match.fieldScores).map(([field, score]) => (
                                  <span key={field} className="inline-block mr-2">
                                    {field}: {(score * 100).toFixed(0)}%
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {match.matchedSegment && (
                            <div>
                              <span className="text-sm font-medium text-gray-700">匹配文本: </span>
                              <span className="text-sm bg-gray-100 px-2 py-1 rounded">
                                {match.matchedSegment}
                              </span>
                            </div>
                          )}
                          
                          <div className="border-t border-gray-200 pt-2">
                            <div className="text-sm text-gray-600">
                              <div>姓名: {match.row.name}</div>
                              <div>地址: {match.row.address}</div>
                              <div>邮箱: {match.row.email}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}