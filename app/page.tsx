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
  matchedSegment: string; // 添加匹配的文本片段
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

const normalizeString = (str: string): string => {
  return str
    .toLowerCase()
    .trim()
    .replace(/\s+/g, '') // 移除所有空格
    .replace(/[^\w\u4e00-\u9fff]/g, ''); // 只保留字母数字和中文字符
};

const calculateSimilarity = (str1: string, str2: string): number => {
  const cleanStr1 = normalizeString(str1);
  const cleanStr2 = normalizeString(str2);
  
  if (!cleanStr1 || !cleanStr2) return 0;
  if (cleanStr1 === cleanStr2) return 1;
  
  // 检查包含关系
  if (cleanStr1.includes(cleanStr2) || cleanStr2.includes(cleanStr1)) return 0.85;
  
  // 检查部分匹配（适合中文姓名）
  if (cleanStr1.length >= 2 && cleanStr2.length >= 2) {
    // 中文姓名部分匹配
    if (cleanStr1.substring(0, 2) === cleanStr2.substring(0, 2)) return 0.7;
  }
  
  // Levenshtein距离相似度
  const distance = calculateLevenshteinDistance(cleanStr1, cleanStr2);
  const maxLength = Math.max(cleanStr1.length, cleanStr2.length);
  const levenshteinSimilarity = maxLength === 0 ? 0 : 1 - distance / maxLength;
  
  // Jaccard相似度（字符集合交集）
  const set1 = new Set(cleanStr1.split(''));
  const set2 = new Set(cleanStr2.split(''));
  const intersection = new Set([...set1].filter(x => set2.has(x)));
  const union = new Set([...set1, ...set2]);
  const jaccardSimilarity = union.size === 0 ? 0 : intersection.size / union.size;
  
  // 综合相似度（加权平均）
  return (levenshteinSimilarity * 0.7 + jaccardSimilarity * 0.3);
};

// 找到最佳匹配的子字符串
const findBestSubstring = (text: string, target: string): string => {
  const normalizedText = normalizeString(text);
  const normalizedTarget = normalizeString(target);
  
  if (normalizedText.includes(normalizedTarget)) {
    // 找到完全匹配的位置
    const startIndex = normalizedText.indexOf(normalizedTarget);
    // 从原始文本中找到对应的位置
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
  
  // 如果没有完全匹配，返回空字符串
  return '';
};

// 增强的匹配函数，支持多字段匹配，对地址字段要求完整匹配
const findBestMatch = (ocrText: string, csvRow: CSVRow): {similarity: number, matchedFields: string[], matchedSegment: string} => {
  const fields = ['name', 'address', 'email'];
  let bestSimilarity = 0;
  const matchedFields: string[] = [];
  let matchedSegment = '';
  
  // 分割OCR文本为多个可能的匹配片段
  const ocrSegments = ocrText.split(/[\s\n\r,，。；;]+/).filter(seg => seg.trim().length > 1);
  
  fields.forEach(field => {
    const csvValue = csvRow[field];
    if (!csvValue) return;
    
    let fieldBestSimilarity = 0;
    let fieldMatched = false;
    let fieldMatchedSegment = '';
    
    if (field === 'address') {
      // 地址字段只需要匹配前8个字符
      const csvAddressPrefix = normalizeString(csvValue).substring(0, 8);
      const ocrNormalized = normalizeString(ocrText);
      
      // 检查OCR文本是否包含地址的前8个字符
      let addressPrefixSimilarity = 0;
      
      if (csvAddressPrefix.length > 0) {
        // 直接包含检查
        if (ocrNormalized.includes(csvAddressPrefix)) {
          addressPrefixSimilarity = 0.9; // 完全包含给高分
          // 只提取完全匹配的字符部分
          const startIndex = ocrNormalized.indexOf(csvAddressPrefix);
          // 从原始OCR文本中找到对应的位置
          let originalStartIndex = 0;
          let normalizedIndex = 0;
          for (let i = 0; i < ocrText.length; i++) {
            const normalizedChar = normalizeString(ocrText[i]);
            if (normalizedIndex === startIndex) {
              originalStartIndex = i;
              break;
            }
            normalizedIndex += normalizedChar.length;
          }
          // 提取完全匹配的字符
          fieldMatchedSegment = ocrText.substring(originalStartIndex, originalStartIndex + csvAddressPrefix.length);
        } else {
          // 计算前8个字符的相似度
          addressPrefixSimilarity = calculateSimilarity(ocrNormalized, csvAddressPrefix);
          if (addressPrefixSimilarity > 0.7) {
            // 找到最相似的文本片段，但只显示匹配的部分
            const bestMatch = findBestSubstring(ocrText, csvAddressPrefix);
            fieldMatchedSegment = bestMatch;
          }
        }
      }
      
      fieldBestSimilarity = addressPrefixSimilarity;
      
      // 地址前缀匹配阈值降低到70%
      if (addressPrefixSimilarity > 0.7) {
        fieldMatched = true;
      }
      
      console.log(`Address prefix matching - OCR: "${ocrText}" vs CSV prefix: "${csvAddressPrefix}" (from "${csvValue}") = ${Math.round(addressPrefixSimilarity * 100)}%`);
      
    } else {
      // 姓名和邮箱字段可以使用部分匹配
      
      // 直接匹配整个OCR文本
      const directSimilarity = calculateSimilarity(ocrText, csvValue);
      if (directSimilarity > fieldBestSimilarity) {
        fieldBestSimilarity = directSimilarity;
        if (directSimilarity > 0.6) {
          // 只显示完全匹配的部分
          const bestMatch = findBestSubstring(ocrText, csvValue);
          fieldMatchedSegment = bestMatch || ocrText.substring(0, Math.min(10, ocrText.length));
        }
      }
      if (directSimilarity > 0.6) {
        fieldMatched = true;
      }
      
      // 匹配OCR文本片段（只对非地址字段）
      ocrSegments.forEach(segment => {
        const segmentSimilarity = calculateSimilarity(segment, csvValue);
        if (segmentSimilarity > fieldBestSimilarity) {
          fieldBestSimilarity = segmentSimilarity;
          if (segmentSimilarity > 0.6) {
            // 只显示完全匹配的部分
            const bestMatch = findBestSubstring(segment, csvValue);
            fieldMatchedSegment = bestMatch || segment;
          }
        }
        if (segmentSimilarity > 0.6) {
          fieldMatched = true;
        }
      });
    }
    
    // 更新最佳相似度和匹配片段
    if (fieldBestSimilarity > bestSimilarity) {
      bestSimilarity = fieldBestSimilarity;
      if (fieldMatchedSegment) {
        matchedSegment = fieldMatchedSegment;
      }
    }
    
    // 添加匹配字段
    if (fieldMatched && !matchedFields.includes(field)) {
      matchedFields.push(field);
    }
  });
  
  // 如果没有找到匹配片段，使用OCR文本的开头部分
  if (!matchedSegment && bestSimilarity > 0.5) {
    matchedSegment = ocrText.substring(0, Math.min(50, ocrText.length)).trim();
  }
  
  return { similarity: bestSimilarity, matchedFields, matchedSegment };
};

const CSVOCRDemo: React.FC = () => {
  // State management
  const [csvData, setCsvData] = useState<CSVRow[]>([]);
  const [csvFileInfo, setCsvFileInfo] = useState<CSVFileInfo | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [ocrText, setOcrText] = useState('');
  const [matchResults, setMatchResults] = useState<MatchResult[]>([]);
  const [isProcessingOCR, setIsProcessingOCR] = useState(false);
  const [confidence, setConfidence] = useState(0);
  const [videoReady, setVideoReady] = useState(false);
  const [pendingCameraStart, setPendingCameraStart] = useState(false);
  const [realTimeOCR, setRealTimeOCR] = useState(false);
  const [ocrInterval, setOcrInterval] = useState<NodeJS.Timeout | null>(null);
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const ocrWorkerRef = useRef<any>(null);

  // Load cached CSV file on component mount
  useEffect(() => {
    const loadCachedCSV = () => {
      try {
        const cached = localStorage.getItem('csvFileInfo');
        if (cached) {
          const fileInfo: CSVFileInfo = JSON.parse(cached);
          setCsvFileInfo(fileInfo);
          setCsvData(fileInfo.data);
          console.log('Loaded cached CSV file:', fileInfo.name);
        }
      } catch (error) {
        console.error('Error loading cached CSV:', error);
        localStorage.removeItem('csvFileInfo');
      }
    };
    
    loadCachedCSV();
  }, []);

  // Initialize OCR worker and cleanup on unmount
  useEffect(() => {
    const initWorker = async () => {
      ocrWorkerRef.current = await createWorker('eng+chi_sim', undefined, {
        logger: m => {
          if (m.status === 'recognizing text') {
            console.log('OCR Progress:', Math.round(m.progress * 100) + '%');
          }
        }
      });
    };
    initWorker();
    
    return () => {
      // Cleanup OCR worker
      if (ocrWorkerRef.current) {
        ocrWorkerRef.current.terminate();
      }
      // Cleanup camera stream
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [cameraStream]);

  // Handle camera initialization when video element is ready
  useEffect(() => {
    if (pendingCameraStart && videoRef.current && isCameraActive && !cameraStream) {
      const initCamera = async () => {
        try {
                     const stream = await navigator.mediaDevices.getUserMedia({
             video: { 
               width: { ideal: 640, max: 1280 },
               height: { ideal: 480, max: 720 },
               facingMode: 'environment' // 使用后置摄像头
             }
           });
          setCameraStream(stream);
          
          if (videoRef.current) {
            console.log('Setting stream to video element', stream);
            videoRef.current.srcObject = stream;
            
            // 确保视频开始播放
            videoRef.current.onloadedmetadata = () => {
              console.log('Video metadata loaded, starting playback');
              setVideoReady(true);
              videoRef.current?.play().catch((playError) => {
                console.error('Error playing video:', playError);
              });
            };
            
            // 添加更多事件监听器用于调试
            videoRef.current.oncanplay = () => {
              console.log('Video can play');
              setVideoReady(true);
            };
            videoRef.current.onplaying = () => console.log('Video is playing');
            videoRef.current.onerror = (e) => console.error('Video error:', e);
          }
          setPendingCameraStart(false);
        } catch (error: any) {
          console.error('Error accessing camera:', error);
          setIsCameraActive(false);
          setPendingCameraStart(false);
          if (error.name === 'NotAllowedError') {
            alert('摄像头权限被拒绝，请在浏览器设置中允许摄像头访问');
          } else if (error.name === 'NotFoundError') {
            alert('未找到摄像头设备');
          } else {
            alert('无法访问摄像头：' + (error.message || '未知错误'));
          }
        }
      };
      
      initCamera();
    }
  }, [pendingCameraStart, isCameraActive, cameraStream, videoRef.current]);

  // Keyboard event handler for manual OCR
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !event.repeat) {
        event.preventDefault(); // 防止页面滚动
        if (videoReady && !isProcessingOCR && !realTimeOCR) {
          captureAndProcessOCR(false);
        }
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => {
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, [videoReady, isProcessingOCR, realTimeOCR]);

  // CSV Upload Handler
  const handleFileUpload = useCallback((file: File) => {
    if (!file) {
      alert('Please select a valid file');
      return;
    }
    
    if (!file.type.includes('csv') && !file.name.toLowerCase().endsWith('.csv')) {
      alert('Please upload a CSV file');
      return;
    }

    setIsUploading(true);
    
    // Use FileReader to read the file content first
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const csvContent = e.target?.result as string;
        Papa.parse(csvContent, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const data = results.data as CSVRow[];
            // Normalize column names
            const normalizedData = data.map(row => {
              const normalizedRow: CSVRow = { name: '', address: '', email: '' };
              
              Object.keys(row).forEach(key => {
                const lowerKey = key.toLowerCase();
                if (lowerKey.includes('name') || lowerKey.includes('姓名')) {
                  normalizedRow.name = row[key] || '';
                } else if (lowerKey.includes('address') || lowerKey.includes('地址')) {
                  normalizedRow.address = row[key] || '';
                } else if (lowerKey.includes('email') || lowerKey.includes('邮箱') || lowerKey.includes('编号')) {
                  normalizedRow.email = row[key] || '';
                }
                normalizedRow[key] = row[key];
              });
              
              return normalizedRow;
            });
            
            const filteredData = normalizedData.filter(row => row.name || row.address || row.email);
            
            // Create file info object
            const fileInfo: CSVFileInfo = {
              name: file.name,
              uploadTime: new Date().toLocaleString('zh-CN'),
              size: file.size,
              data: filteredData
            };
            
            // Save to state and localStorage
            setCsvData(filteredData);
            setCsvFileInfo(fileInfo);
            
            // Cache in localStorage
            try {
              localStorage.setItem('csvFileInfo', JSON.stringify(fileInfo));
              console.log('CSV file cached:', fileInfo.name);
            } catch (error) {
              console.error('Error caching CSV file:', error);
            }
            
            setIsUploading(false);
          },
          error: (error: any) => {
            console.error('CSV parsing error:', error);
            alert('Error parsing CSV file');
            setIsUploading(false);
          }
        });
      } catch (error) {
        console.error('File reading error:', error);
        alert('Error reading file');
        setIsUploading(false);
      }
    };
    
    reader.onerror = () => {
      console.error('FileReader error');
      alert('Error reading file');
      setIsUploading(false);
    };
    
    reader.readAsText(file);
  }, []);

  // File input handler only (removed drag and drop)

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
    // Reset input value to allow same file selection again
    e.target.value = '';
  }, [handleFileUpload]);

  // Camera handlers  
  const startCamera = () => {
    console.log('Starting camera - setting states');
    setIsCameraActive(true);
    setPendingCameraStart(true);
    setVideoReady(false);
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    // 停止实时OCR
    if (ocrInterval) {
      clearInterval(ocrInterval);
      setOcrInterval(null);
    }
    setIsCameraActive(false);
    setVideoReady(false);
    setPendingCameraStart(false);
    setRealTimeOCR(false);
  };

  // OCR Processing
  const captureAndProcessOCR = async (isRealTime = false) => {
    if (!videoRef.current || !canvasRef.current || !ocrWorkerRef.current) return;
    
    // 如果是实时模式且正在处理，则跳过这次识别
    if (isRealTime && isProcessingOCR) return;
    
    setIsProcessingOCR(true);
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    if (ctx) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      
      try {
        const { data: { text, confidence } } = await ocrWorkerRef.current.recognize(canvas);
        setOcrText(text);
        setConfidence(confidence);
        
        // Process matches only if there's text and CSV data
        if (text && text.trim() && csvData.length > 0) {
          const matches = findMatches(text.trim());
          setMatchResults(matches);
        }
        
        if (isRealTime) {
          console.log('Real-time OCR:', text.slice(0, 50) + '...', 'Confidence:', Math.round(confidence));
        }
      } catch (error) {
        console.error('OCR Error:', error);
      }
    }
    
    setIsProcessingOCR(false);
  };

  // Real-time OCR toggle
  const toggleRealTimeOCR = () => {
    if (realTimeOCR) {
      // Stop real-time OCR
      if (ocrInterval) {
        clearInterval(ocrInterval);
        setOcrInterval(null);
      }
      setRealTimeOCR(false);
    } else {
      // Start real-time OCR
      if (videoReady && ocrWorkerRef.current) {
        setRealTimeOCR(true);
        const interval = setInterval(() => {
          captureAndProcessOCR(true);
        }, 2000); // 每2秒识别一次
        setOcrInterval(interval);
      }
    }
  };

  // Enhanced match finding logic with strict address matching
  const findMatches = (text: string): MatchResult[] => {
    const matches: MatchResult[] = [];
    
    console.log('Finding matches for OCR text:', text);
    
    csvData.forEach((row, index) => {
      const matchResult = findBestMatch(text, row);
      
      // 不同字段使用不同的阈值
      let shouldMatch = false;
      
      if (matchResult.matchedFields.includes('address')) {
        // 如果匹配到地址字段（前8个字符），需要70%以上的相似度
        shouldMatch = matchResult.similarity >= 0.7;
      } else if (matchResult.matchedFields.length > 0) {
        // 其他字段匹配，使用较低阈值
        shouldMatch = matchResult.similarity >= 0.5;
      }
      
      if (shouldMatch) {
        matches.push({
          row,
          matchedFields: matchResult.matchedFields,
          confidence: confidence,
          similarity: matchResult.similarity,
          ocrText: text,
          matchedSegment: matchResult.matchedSegment
        });
        
        console.log(`Match found for row ${index}:`, {
          similarity: matchResult.similarity,
          matchedFields: matchResult.matchedFields,
          name: row.name,
          address: row.address,
          email: row.email,
          threshold: matchResult.matchedFields.includes('address') ? '70% (前8字符)' : '50%'
        });
      }
    });
    
    // 按相似度排序，优先显示地址匹配的结果
    return matches.sort((a, b) => {
      // 地址匹配的结果优先排序
      const aHasAddress = a.matchedFields.includes('address');
      const bHasAddress = b.matchedFields.includes('address');
      
      if (aHasAddress && !bHasAddress) return -1;
      if (!aHasAddress && bHasAddress) return 1;
      
      // 相同类型的匹配按相似度排序
      return b.similarity - a.similarity;
    }).slice(0, 8);
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">PO Box匹配系统</h1>
      </div>

      {/* 响应式布局：左侧整体(上传+扫描) + 右侧整体(结果) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* 左侧整体：上传文件模块 + 摄像头扫描模块 */}
        <div className="space-y-8">
          {/* 模块1: CSV文件上传 - 增强版 */}
          <Card className="max-w-md mx-auto lg:mx-0">
            <CardContent className="p-4">
              <p className="text-gray-600 text-sm mb-3 text-center leading-tight font-sans">上传CSV文件，使用摄像头识别文字并匹配PO Box数据</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              onChange={handleFileInputChange}
              className="hidden"
            />

            {/* 文件信息显示 */}
            {csvFileInfo ? (
              <div className="py-1">
                <div className="flex items-center justify-center gap-4 mb-1">
                  <span className="text-sm font-normal text-gray-800 font-sans leading-tight">{csvFileInfo.name}</span>
                  <span className="text-sm font-normal text-gray-500 font-sans leading-tight">
                    {csvFileInfo.uploadTime}
                  </span>
                  <span className="text-sm font-normal text-gray-700 font-sans leading-tight">
                    {csvFileInfo.data.length} 条记录已加载
                  </span>
                </div>
                <div className="flex justify-center gap-2">
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-1 px-3 text-sm font-sans"
                  >
                    <Upload className="w-3 h-3" />
                    {isUploading ? '上传中...' : '选择文件'}
                  </Button>
                  <Button
                    onClick={() => {
                      localStorage.removeItem('csvFileInfo');
                      setCsvFileInfo(null);
                      setCsvData([]);
                    }}
                    variant="outline"
                    size="sm"
                    className="px-3 text-gray-600 hover:text-gray-800 hover:bg-gray-100 text-sm font-sans"
                  >
                    清除缓存
                  </Button>
                </div>
              </div>
                          ) : (
                <div className="flex items-center justify-center py-3">
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                    size="sm"
                    className="flex items-center gap-2 text-sm font-sans"
                  >
                    <Upload className="w-4 h-4" />
                    {isUploading ? '上传中...' : '选择CSV文件'}
                  </Button>
                </div>
              )}

            {/* 兼容旧版本显示 */}
            {csvData.length > 0 && !csvFileInfo && (
              <div className="mt-3">
                <div className="text-center">
                  <span className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                    ✓ {csvData.length} 条记录已加载
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

          {/* 模块2: 摄像头扫描 */}
          <Card className="max-w-md mx-auto lg:mx-0">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="w-5 h-5" />
                摄像头扫描
              </CardTitle>
            </CardHeader>
          <CardContent>
            {!isCameraActive ? (
              <div className="text-center py-8">
                <Camera className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                <p className="text-lg mb-4">点击开启摄像头开始扫描</p>
                <Button onClick={startCamera}>
                  <Eye className="w-4 h-4 mr-2" />
                  开启摄像头
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    controls={false}
                    className="w-full h-auto max-w-sm mx-auto rounded-lg border-2 border-gray-200 bg-black object-cover"
                    style={{ aspectRatio: '4/3', minHeight: '200px' }}
                  />
                  <canvas
                    ref={canvasRef}
                    className="hidden"
                  />
                  {isCameraActive && (
                    <div className="absolute top-2 left-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                      录制中
                    </div>
                  )}
                  {realTimeOCR && (
                    <div className="absolute top-2 right-2 bg-blue-500 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                      实时识别
                    </div>
                  )}
                  {!videoReady && isCameraActive && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg">
                      <div className="text-white text-sm">正在加载摄像头...</div>
                    </div>
                  )}
                </div>
                
                {/* 摄像头状态显示 - 已隐藏 */}
                {/* <div className="text-center text-xs text-gray-500 space-y-1">
                  <div>摄像头状态: {isCameraActive ? '已启动' : '未启动'}</div>
                  <div>视频流状态: {videoReady ? '已就绪' : '未就绪'}</div>
                  <div>实时识别: {realTimeOCR ? '运行中' : '已停止'}</div>
                  {realTimeOCR && (
                    <div className="text-xs bg-yellow-50 text-yellow-700 px-2 py-1 rounded border">
                      💡 地址匹配前8个字符(70%+)，其他字段部分匹配即可
                    </div>
                  )}
                  {isCameraActive && !videoReady && (
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => {
                        if (videoRef.current) {
                          videoRef.current.play().catch(console.error);
                        }
                      }}
                    >
                      手动播放视频
                    </Button>
                  )}
                </div> */}
                
                <div className="flex justify-center gap-1">
                  <Button
                    onClick={() => captureAndProcessOCR(false)}
                    disabled={isProcessingOCR || realTimeOCR}
                    variant="outline"
                    size="sm"
                    className="text-xs px-2 py-1 h-8"
                  >
                    <Play className="w-3 h-3 mr-1" />
                    手动识别
                  </Button>
                  <Button
                    onClick={toggleRealTimeOCR}
                    disabled={!videoReady}
                    variant={realTimeOCR ? "default" : "outline"}
                    size="sm"
                    className="text-xs px-2 py-1 h-8"
                  >
                    {realTimeOCR ? (
                      <>
                        <Square className="w-3 h-3 mr-1" />
                        停止识别
                      </>
                    ) : (
                      <>
                        <Play className="w-3 h-3 mr-1" />
                        实时识别
                      </>
                    )}
                  </Button>
                  <Button 
                    onClick={stopCamera} 
                    variant="outline" 
                    size="sm"
                    className="text-xs px-2 py-1 h-8"
                  >
                    <EyeOff className="w-3 h-3 mr-1" />
                    关闭
                  </Button>
                </div>

                {/* 识别结果已隐藏，只在匹配结果中显示 */}
              </div>
            )}
          </CardContent>
        </Card>
        </div>

        {/* 右侧整体：匹配结果模块 */}
        <div>
          {/* 模块3: 匹配结果 */}
          <Card className="max-w-md mx-auto lg:mx-0">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Eye className="w-5 h-5" />
                  匹配结果
                </div>
                {matchResults.length > 0 && (
                  <div className="text-right">
                    <div className="text-2xl font-bold text-red-600">
                      PO Box: {matchResults[0].row.email || matchResults[0].row['PO Box'] || matchResults[0].row['po box'] || matchResults[0].row.pobox || '—'}
                    </div>
                  </div>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {matchResults.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <p>暂无匹配结果</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {matchResults.map((match, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                      {/* 匹配信息头部 - 已隐藏 */}
                      {/* <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 bg-gray-700 text-white rounded-full flex items-center justify-center text-sm font-bold">
                            {index + 1}
                          </div>
                          <div>
                            <div className="text-sm text-gray-800 font-medium">
                              匹配度: {Math.round(match.similarity * 100)}%
                            </div>
                            <div className="text-xs text-gray-500">
                              OCR置信度: {Math.round(match.confidence)}%
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
                            匹配字段: {match.matchedFields.join(', ') || '综合匹配'}
                          </div>
                        </div>
                      </div> */}

                      {/* 匹配文本显示框 */}
                      {match.matchedSegment && (
                        <div className="mb-3 p-2 bg-white border border-gray-200 rounded text-sm">
                          <span className="text-gray-700 font-medium">匹配文本: </span>
                          <span className="font-mono text-gray-800 bg-gray-100 px-1 rounded">
                            {match.matchedSegment}
                          </span>
                        </div>
                      )}

                      {/* 识别文本显示 - 已隐藏 */}
                      {/* <div className="mb-3 p-2 bg-gray-100 rounded text-sm">
                        <span className="text-gray-600">识别文本: </span>
                        <span className="font-mono">{match.ocrText}</span>
                      </div> */}

                      {/* CSV完整记录 */}
                      <div className="bg-white rounded p-3 border-l-4 border-gray-400">
                        <div className="grid grid-cols-1 gap-2 text-sm">
                          <div className="flex">
                            <span className="w-20 text-gray-600 font-medium">Name:</span>
                            <span className="text-gray-800">{match.row.name || '—'}</span>
                          </div>
                          <div className="flex">
                            <span className="w-20 text-gray-600 font-medium">Address:</span>
                            <span className="text-gray-800">{match.row.address || '—'}</span>
                          </div>
                          <div className="flex">
                            <span className="w-20 text-gray-600 font-medium">PO Box:</span>
                            <span className="text-gray-800">{match.row.email || match.row['PO Box'] || match.row['po box'] || match.row.pobox || '—'}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {/* 匹配统计 */}
                  <div className="text-center text-xs text-gray-600 bg-gray-100 p-2 rounded border border-gray-200">
                    共找到 {matchResults.length} 条匹配记录
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

      </div>
    </div>
  );
};

export default CSVOCRDemo;