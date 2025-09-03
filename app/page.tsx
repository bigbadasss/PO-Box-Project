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
  serviceDescription: string;
  customerName: string;
  streetNumber: string;
  suburb: string;
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
    .replace(/\s+/g, ' ') // 保留单个空格，不完全移除
    .replace(/[^\w\s\u4e00-\u9fff-]/g, ''); // 保留空格和连字符，移除其他特殊字符
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

// 计算更准确的匹配字符数
const calculateMatchingChars = (str1: string, str2: string): number => {
  const cleanStr1 = normalizeString(str1);
  const cleanStr2 = normalizeString(str2);
  
  // 使用动态规划计算最长公共子序列长度
  const dp = Array(cleanStr1.length + 1).fill(null).map(() => Array(cleanStr2.length + 1).fill(0));
  
  for (let i = 1; i <= cleanStr1.length; i++) {
    for (let j = 1; j <= cleanStr2.length; j++) {
      if (cleanStr1[i-1] === cleanStr2[j-1]) {
        dp[i][j] = dp[i-1][j-1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
      }
    }
  }
  
  return dp[cleanStr1.length][cleanStr2.length];
};

// 提取地址中的数字部分和街道名称部分
const parseAddress = (address: string): {numbers: string, streetName: string} => {
  const normalized = normalizeString(address);
  // 匹配开头的数字部分
  const numberMatch = normalized.match(/^[\d\s-]+/);
  const numbers = numberMatch ? numberMatch[0].trim() : '';
  const streetName = normalized.substring(numbers.length).trim();
  
  return { numbers, streetName };
};

// 提取关键单词（优先数字后的第一个单词，如果没有数字则取第一个有意义的单词）
const extractFirstKeyWord = (text: string): string => {
  const normalized = normalizeString(text);
  
  // 优先匹配数字后的第一个有意义单词（长度≥3）
  const afterNumberMatch = normalized.match(/\d+\s+([a-zA-Z\u4e00-\u9fff]{3,})/);
  if (afterNumberMatch) {
    return afterNumberMatch[1];
  }
  
  // 如果没有数字，取第一个有意义的单词（长度≥3）
  const firstWordMatch = normalized.match(/([a-zA-Z\u4e00-\u9fff]{3,})/);
  return firstWordMatch ? firstWordMatch[1] : '';
};

// 从PO Box字符串中提取数字
const extractPOBoxNumber = (poBoxText: string): string => {
  if (!poBoxText) return '';
  
  // 匹配PO Box后面的数字
  const match = poBoxText.match(/(?:PO\s*Box\s*|pobox\s*|po\s*box\s*)(\d+)/i);
  if (match && match[1]) {
    return match[1];
  }
  
  // 如果没有找到PO Box格式，但是字符串包含数字，返回第一个数字
  const numberMatch = poBoxText.match(/\d+/);
  return numberMatch ? numberMatch[0] : poBoxText;
};

// 移除地址中的常见后缀词
const removeAddressSuffixes = (address: string): string => {
  if (!address) return '';
  
  // 常见的道路/地址后缀词（英文）
  const suffixes = [
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'drive', 'dr', 
    'lane', 'ln', 'way', 'close', 'court', 'ct', 'place', 'pl',
    'crescent', 'cres', 'terrace', 'tce', 'circuit', 'cct',
    'boulevard', 'blvd', 'highway', 'hwy', 'esplanade', 'esp',
    'parade', 'pde', 'grove', 'gr', 'walk', 'gardens', 'gdns'
  ];
  
  const normalized = normalizeString(address);
  const words = normalized.split(' ').filter(word => word.length > 0);
  
  // 如果最后一个词是后缀词，移除它
  if (words.length > 1 && suffixes.includes(words[words.length - 1])) {
    const filteredWords = words.slice(0, -1);
    console.log(`  地址后缀过滤: "${address}" -> "${filteredWords.join(' ')}" (移除了"${words[words.length - 1]}")`);
    return filteredWords.join(' ');
  }
  
  return normalized;
};

// 街道号码+地址匹配算法 - 专门匹配 "34 ALF CASEY ROAD" 格式
const findBestMatch = (ocrText: string, csvRow: CSVRow): {similarity: number, matchedFields: string[], matchedSegment: string} => {
  let bestSimilarity = 0;
  const matchedFields: string[] = [];
  let matchedSegment = '';
  
  // 检查必要字段 - 至少要有地址
  if (!csvRow.address) {
    return { similarity: 0, matchedFields: [], matchedSegment: '' };
  }
  
  const ocrNormalized = normalizeString(ocrText);
  console.log(`=== 街道号码+地址匹配 ===`);
  console.log(`OCR文本: "${ocrText}" -> 标准化: "${ocrNormalized}"`);
  console.log(`CSV数据: 门牌号="${csvRow.streetNumber}", 地址="${csvRow.address}"`);
  
  // 1. 提取OCR中的数字和地址部分
  const ocrNumbers = ocrText.match(/^\d+/)?.[0] || ''; // 开头的数字
  const ocrAddressPart = ocrText.replace(/^\d+\s*/, '').trim(); // 移除开头数字后的部分
  
  console.log(`  OCR解析: 数字="${ocrNumbers}", 地址部分="${ocrAddressPart}"`);
  
  // 2. 构建CSV完整地址用于比较
  const csvFullAddress = csvRow.streetNumber ? `${csvRow.streetNumber} ${csvRow.address}`.trim() : csvRow.address;
  const csvFullAddressNormalized = normalizeString(csvFullAddress);
  const csvAddressNormalized = normalizeString(csvRow.address);
  
  console.log(`  CSV完整地址: "${csvFullAddress}" -> 标准化: "${csvFullAddressNormalized}"`);
  
  // 3. 门牌号/街道前缀匹配检查
  let numberMatchScore = 0;
  const csvStreetNumber = csvRow.streetNumber ? csvRow.streetNumber.trim() : '';
  const isCSVNumberDigit = /^\d+$/.test(csvStreetNumber); // 检查CSV中是否为纯数字
  
  if (ocrNumbers && csvStreetNumber && isCSVNumberDigit) {
    // 情况1：OCR有数字，CSV也有数字门牌号 - 严格匹配
    if (ocrNumbers === csvStreetNumber) {
      numberMatchScore = 1.0;
      console.log(`  ✅ 数字门牌号完全匹配: "${ocrNumbers}"`);
    } else {
      numberMatchScore = 0.1;
      console.log(`  ❌ 数字门牌号不匹配: OCR"${ocrNumbers}" vs CSV"${csvStreetNumber}" - 严格模式拒绝`);
    }
  } else if (ocrNumbers && csvStreetNumber && !isCSVNumberDigit) {
    // 情况2：OCR有数字，但CSV的streetNumber是文字（如"GLEN"） - 降级为地址匹配
    numberMatchScore = 0.7; // 给较高基础分，让地址匹配决定
    console.log(`  ⚠️  OCR有数字"${ocrNumbers}"但CSV streetNumber是文字"${csvStreetNumber}" - 降级为地址匹配`);
  } else if (!ocrNumbers && csvStreetNumber && !isCSVNumberDigit) {
    // 情况3：OCR无数字，CSV streetNumber是文字 - 这种情况需要匹配街道前缀
    const ocrNormalized = normalizeString(ocrText);
    const csvStreetNormalized = normalizeString(csvStreetNumber);
    
    // 检查OCR文本开头是否包含街道前缀
    const ocrWords = ocrNormalized.split(' ').filter(w => w.length >= 2);
    const ocrStartsWithPrefix = ocrWords.length > 0 && calculateSimilarity(ocrWords[0], csvStreetNormalized) >= 0.8;
    
    if (ocrStartsWithPrefix) {
      numberMatchScore = 0.95; // 开头匹配给最高分
      console.log(`  ✅ 街道前缀完美匹配: OCR开头"${ocrWords[0]}" ≈ CSV前缀"${csvStreetNumber}"`);
    } else if (ocrNormalized.includes(csvStreetNormalized) || csvStreetNormalized.includes(ocrNormalized)) {
      numberMatchScore = 0.8; // 包含匹配给中等分
      console.log(`  ✅ 街道前缀包含匹配: OCR包含"${csvStreetNumber}"`);
    } else {
      numberMatchScore = 0.4; // 给较低基础分让地址匹配决定
      console.log(`  ⚠️  街道前缀未匹配，依赖地址匹配 (OCR:"${ocrWords[0]}" vs CSV:"${csvStreetNumber}")`);
    }
  } else if (!ocrNumbers && (!csvStreetNumber || csvStreetNumber === '')) {
    numberMatchScore = 1.0; // 都没有门牌号，认为匹配
    console.log(`  ✅ 都无门牌号，允许匹配`);
  } else if (ocrNumbers && (!csvStreetNumber || csvStreetNumber === '')) {
    // OCR有门牌号但CSV没有 - 降级到地址匹配
    numberMatchScore = 0.5; 
    console.log(`  ⚠️  OCR有门牌号"${ocrNumbers}"但CSV无streetNumber，降级处理`);
  } else {
    numberMatchScore = 0.3;
    console.log(`  ⚠️  其他情况: OCR="${ocrNumbers}", CSV="${csvStreetNumber}"`);
  }
  
  // 4. 地址名称匹配检查 - 移除后缀词后进行匹配
  let addressMatchScore = 0;
  const addressToMatch = ocrAddressPart || ocrText; // 如果没有地址部分，使用整个OCR文本
  
  if (addressToMatch && csvRow.address) {
    // 对OCR和CSV地址都进行后缀词过滤
    const ocrAddressFiltered = removeAddressSuffixes(addressToMatch);
    const csvAddressFiltered = removeAddressSuffixes(csvRow.address);
    
    console.log(`  地址过滤后比较: OCR"${ocrAddressFiltered}" vs CSV"${csvAddressFiltered}"`);
    
    // 完全包含检查
    if (csvAddressFiltered.includes(ocrAddressFiltered)) {
      addressMatchScore = 0.95;
      console.log(`  ✅ 地址完全包含: "${ocrAddressFiltered}" 在 "${csvAddressFiltered}" 中找到`);
    } else if (ocrAddressFiltered.includes(csvAddressFiltered)) {
      addressMatchScore = 0.9;
      console.log(`  ✅ 地址反向包含: "${csvAddressFiltered}" 在 "${ocrAddressFiltered}" 中找到`);
    } else {
      // 单词级别匹配 - 使用过滤后的地址，优先考虑连续词组匹配
      const ocrWords = ocrAddressFiltered.split(' ').filter(w => w.length >= 2);
      const csvWords = csvAddressFiltered.split(' ').filter(w => w.length >= 2);
      
      // 1. 检查连续词组匹配（如 "alpine station"）
      let sequenceBonus = 0;
      for (let i = 0; i < ocrWords.length - 1; i++) {
        const ocrSequence = ocrWords[i] + ' ' + ocrWords[i + 1];
        for (let j = 0; j < csvWords.length - 1; j++) {
          const csvSequence = csvWords[j] + ' ' + csvWords[j + 1];
          if (calculateSimilarity(ocrSequence, csvSequence) >= 0.9) {
            sequenceBonus += 0.3; // 连续词组匹配奖励
            console.log(`    ✅ 连续词组匹配: "${ocrSequence}" ≈ "${csvSequence}"`);
          }
        }
      }
      
      // 2. 单词级别匹配
      let matchedWords = 0;
      let bestWordMatches: string[] = [];
      
      ocrWords.forEach(ocrWord => {
        let bestMatch = '';
        let bestSimilarity = 0;
        
        csvWords.forEach(csvWord => {
          const similarity = calculateSimilarity(ocrWord, csvWord);
          if (similarity >= 0.8 && similarity > bestSimilarity) {
            bestSimilarity = similarity;
            bestMatch = csvWord;
          }
        });
        
        if (bestMatch) {
          matchedWords++;
          bestWordMatches.push(`"${ocrWord}" ≈ "${bestMatch}" (${Math.round(bestSimilarity * 100)}%)`);
        }
      });
      
      if (ocrWords.length > 0) {
        const baseScore = (matchedWords / ocrWords.length) * 0.85;
        addressMatchScore = Math.min(0.95, baseScore + sequenceBonus); // 加上连续匹配奖励，最高95%
        console.log(`  地址单词匹配(过滤后): ${matchedWords}/${ocrWords.length} = ${Math.round(baseScore * 100)}% + 连续奖励${Math.round(sequenceBonus * 100)}% = ${Math.round(addressMatchScore * 100)}%`);
        bestWordMatches.forEach(match => console.log(`    ✅ ${match}`));
      }
      
      // 如果过滤后的匹配度不高，尝试原始地址匹配作为备用
      if (addressMatchScore < 0.6) {
        const ocrAddressNormalized = normalizeString(addressToMatch);
        const csvAddressNormalized = normalizeString(csvRow.address);
        const backupScore = calculateSimilarity(ocrAddressNormalized, csvAddressNormalized) * 0.7;
        if (backupScore > addressMatchScore) {
          addressMatchScore = backupScore;
          console.log(`  使用备用匹配(未过滤): ${Math.round(backupScore * 100)}%`);
        }
      }
    }
  }
  
  // 5. 整体匹配检查（备用）
  let fullMatchScore = 0;
  if (ocrNormalized && csvFullAddressNormalized) {
    if (csvFullAddressNormalized.includes(ocrNormalized)) {
      fullMatchScore = 0.95;
      console.log(`  ✅ 完整地址包含匹配`);
    } else if (ocrNormalized.includes(csvFullAddressNormalized)) {
      fullMatchScore = 0.9;
      console.log(`  ✅ 完整地址反向包含匹配`);
    } else {
      fullMatchScore = calculateSimilarity(ocrNormalized, csvFullAddressNormalized) * 0.8;
      console.log(`  完整地址相似度: ${Math.round(fullMatchScore * 100)}%`);
    }
  }
  
  // 6. 综合评分策略
  const weights = {
    number: 0.4,        // 门牌号权重
    address: 0.5,       // 地址名权重  
    fullMatch: 0.1      // 整体匹配权重
  };
  
  bestSimilarity = (numberMatchScore * weights.number) + 
                   (addressMatchScore * weights.address) + 
                   (fullMatchScore * weights.fullMatch);
  
  // 7. 匹配判断逻辑
  let isValidMatch = false;
  let matchReason = '';
  
  // 检查是否有门牌号信息
  const hasOcrNumber = ocrNumbers && ocrNumbers.length > 0;
  const hasCsvStreetNumber = csvStreetNumber && csvStreetNumber.length > 0;
  
  if (hasOcrNumber && hasCsvStreetNumber && isCSVNumberDigit) {
    // 情况1：OCR有数字，CSV也有数字门牌号 - 严格验证
    if (ocrNumbers === csvStreetNumber) {
      if (addressMatchScore >= 0.7) {
        isValidMatch = true;
        matchReason = `完整地址匹配: 门牌号100% + 地址${Math.round(addressMatchScore * 100)}%`;
      } else {
        matchReason = `门牌号匹配但地址匹配度不足 (门牌号100%, 地址${Math.round(addressMatchScore * 100)}%)`;
      }
    } else {
      isValidMatch = false;
      matchReason = `数字门牌号不匹配 (OCR:"${ocrNumbers}" vs CSV:"${csvStreetNumber}")`;
    }
  } else if (hasOcrNumber && hasCsvStreetNumber && !isCSVNumberDigit) {
    // 情况2：OCR有数字，CSV street number是文字 - 主要看地址匹配
    if (addressMatchScore >= 0.8) {
      isValidMatch = true;
      matchReason = `地址主导匹配: ${Math.round(addressMatchScore * 100)}% (CSV街道前缀:"${csvStreetNumber}")`;
    } else {
      matchReason = `地址匹配度不足: ${Math.round(addressMatchScore * 100)}% (需要≥80%用于文字前缀情况)`;
    }
  } else if (!hasOcrNumber && hasCsvStreetNumber && !isCSVNumberDigit) {
    // 情况3：无OCR数字，CSV street number是文字 - 最适合Glen Alpine Station的情况
    if (numberMatchScore >= 0.8 && addressMatchScore >= 0.7) {
      isValidMatch = true;
      matchReason = `街道前缀+地址匹配: 前缀${Math.round(numberMatchScore * 100)}% + 地址${Math.round(addressMatchScore * 100)}%`;
    } else if (addressMatchScore >= 0.85) {
      isValidMatch = true;
      matchReason = `强地址匹配: ${Math.round(addressMatchScore * 100)}%`;
    } else {
      matchReason = `匹配度不足: 前缀${Math.round(numberMatchScore * 100)}% + 地址${Math.round(addressMatchScore * 100)}%`;
    }
  } else if (!hasOcrNumber && !hasCsvStreetNumber) {
    // 情况2：都没有门牌号 - 纯地址匹配
    if (addressMatchScore >= 0.85) {
      isValidMatch = true;
      matchReason = `纯地址匹配: ${Math.round(addressMatchScore * 100)}%`;
    } else if (fullMatchScore >= 0.8) {
      isValidMatch = true;
      matchReason = `整体地址匹配: ${Math.round(fullMatchScore * 100)}%`;
    }
  } else {
    // 情况3：门牌号不对称 - 降级匹配
    if (bestSimilarity >= 0.7) {
      isValidMatch = true;
      matchReason = `不对称匹配: 综合${Math.round(bestSimilarity * 100)}%`;
    } else if (addressMatchScore >= 0.9) {
      isValidMatch = true;
      matchReason = `强地址匹配: ${Math.round(addressMatchScore * 100)}%`;
    }
  }
  
  // 兜底：高综合匹配度
  if (!isValidMatch && bestSimilarity >= 0.75) {
    isValidMatch = true;
    matchReason = `高综合匹配: ${Math.round(bestSimilarity * 100)}%`;
  }
  
  if (!isValidMatch) {
    matchReason = `匹配度不足 (门牌号${Math.round(numberMatchScore * 100)}%, 地址${Math.round(addressMatchScore * 100)}%, 综合${Math.round(bestSimilarity * 100)}%)`;
  }
  
  if (isValidMatch) {
    matchedFields.push('streetAddress');
    matchedSegment = ocrText;
  }
  
  console.log(`  门牌号匹配: ${Math.round(numberMatchScore * 100)}%`);
  console.log(`  地址匹配: ${Math.round(addressMatchScore * 100)}%`);
  console.log(`  综合评分: ${Math.round(bestSimilarity * 100)}%`);
  console.log(`  匹配结果: ${isValidMatch} (${matchReason})`);
  console.log(`========================`);
  
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
            alert('Camera access denied. Please allow camera access in browser settings.');
          } else if (error.name === 'NotFoundError') {
            alert('Camera device not found.');
          } else {
            alert('Unable to access camera: ' + (error.message || 'Unknown error'));
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
              const normalizedRow: CSVRow = { 
                name: '', 
                address: '', 
                email: '',
                serviceDescription: '',
                customerName: '',
                streetNumber: '',
                suburb: ''
              };
              
              Object.keys(row).forEach(key => {
                const lowerKey = key.toLowerCase();
                if (lowerKey.includes('service description') || lowerKey === 'service description') {
                  normalizedRow.serviceDescription = row[key] || '';
                  normalizedRow.email = row[key] || ''; // 保持兼容性，PO Box信息显示在email字段
                } else if (lowerKey.includes('customer name') || lowerKey === 'customer name') {
                  normalizedRow.customerName = row[key] || '';
                  normalizedRow.name = row[key] || ''; // 保持兼容性
                } else if (lowerKey.includes('street number') || lowerKey === 'street number') {
                  normalizedRow.streetNumber = row[key] || '';
                  console.log(`解析Street Number: "${key}" -> "${row[key]}" -> normalizedRow.streetNumber: "${normalizedRow.streetNumber}"`);
                } else if (lowerKey.includes('address') && !lowerKey.includes('street')) {
                  normalizedRow.address = row[key] || '';
                } else if (lowerKey.includes('suburb')) {
                  normalizedRow.suburb = row[key] || '';
                }
                // 保存原始数据
                normalizedRow[key] = row[key];
              });
              
              return normalizedRow;
            });
            
            const filteredData = normalizedData.filter(row => 
              row.serviceDescription || row.customerName || row.streetNumber || row.address || row.suburb
            );
            
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
      
      // 创建限定框区域的临时canvas
      const croppedCanvas = document.createElement('canvas');
      const croppedCtx = croppedCanvas.getContext('2d');
      
      if (croppedCtx) {
        // 计算限定框在视频中的实际位置和尺寸
        const videoRect = video.getBoundingClientRect();
        const frameWidth = 168; // 限定框宽度 (12个字母)
        const frameHeight = 28; // 限定框高度 (单行)
        
        // 计算比例和中心位置
        const scaleX = video.videoWidth / videoRect.width;
        const scaleY = video.videoHeight / videoRect.height;
        
        const centerX = video.videoWidth / 2;
        const centerY = video.videoHeight / 2;
        
        const cropX = centerX - (frameWidth * scaleX) / 2;
        const cropY = centerY - (frameHeight * scaleY) / 2;
        const cropWidth = frameWidth * scaleX;
        const cropHeight = frameHeight * scaleY;
        
        // 设置裁剪后的canvas尺寸
        croppedCanvas.width = cropWidth;
        croppedCanvas.height = cropHeight;
        
        // 从原始canvas中裁剪指定区域
        croppedCtx.drawImage(
          canvas,
          Math.max(0, cropX),
          Math.max(0, cropY),
          Math.min(cropWidth, video.videoWidth - Math.max(0, cropX)),
          Math.min(cropHeight, video.videoHeight - Math.max(0, cropY)),
          0,
          0,
          cropWidth,
          cropHeight
        );
        
        try {
          // 使用裁剪后的图像进行OCR识别
          const { data: { text, confidence } } = await ocrWorkerRef.current.recognize(croppedCanvas);
          setOcrText(text);
          setConfidence(confidence);
          
          // Process matches only if there's text and CSV data
          if (text && text.trim() && csvData.length > 0) {
            const matches = findMatches(text.trim());
            setMatchResults(matches);
          }
          
          if (isRealTime) {
            console.log('Real-time OCR (框选区域):', text.slice(0, 50) + '...', 'Confidence:', Math.round(confidence));
          }
        } catch (error) {
          console.error('OCR Error:', error);
        }
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

  // 只匹配地址字段的逻辑
  const findMatches = (text: string): MatchResult[] => {
    const matches: MatchResult[] = [];
    
    console.log('=== 开始查找匹配结果 ===');
    console.log('OCR识别文本:', text);
    
    csvData.forEach((row, index) => {
      const matchResult = findBestMatch(text, row);
      
      // 严格的匹配条件：只有当匹配字段不为空时才算有效匹配
      // 这确保了数字和街道名称都满足阈值要求
      if (matchResult.matchedFields.length > 0) {
        matches.push({
          row,
          matchedFields: matchResult.matchedFields,
          confidence: confidence,
          similarity: matchResult.similarity,
          ocrText: text,
          matchedSegment: matchResult.matchedSegment
        });
        
        console.log(`✅ 找到有效匹配 [行 ${index}]:`, {
          name: row.name,
          address: row.address,
          email: row.email,
          similarity: Math.round(matchResult.similarity * 100) + '%',
          matchedFields: matchResult.matchedFields
        });
      } else {
        console.log(`❌ 无效匹配 [行 ${index}]: ${row.address} (未满足阈值要求)`);
      }
    });
    
    console.log(`=== 匹配完成，找到 ${matches.length} 个有效匹配 ===`);
    
    // 按相似度排序，显示前8个最佳匹配
    const sortedMatches = matches.sort((a, b) => b.similarity - a.similarity).slice(0, 8);
    
    console.log('最终排序结果:');
    sortedMatches.forEach((match, index) => {
      console.log(`${index + 1}. ${match.row.address} (${Math.round(match.similarity * 100)}%)`);
    });
    
    return sortedMatches;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Title - Mobile Optimized */}
      <div className="bg-white shadow-sm border-b px-4 py-4 sm:py-6">
        <h1 className="text-xl sm:text-2xl md:text-3xl font-bold text-center text-gray-800">
          PO Box Matcher
        </h1>
      </div>

      {/* Main Content Area */}
      <div className="px-2 sm:px-4 py-4 sm:py-6 space-y-4 sm:space-y-6">
        
        {/* Top Section: Upload and Scan Modules - Mobile vertical, desktop side-by-side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 max-w-6xl mx-auto">
        
          {/* Left: File Upload Module */}
          <div>
            {/* Module 1: CSV File Upload - Mobile Optimized */}
            <Card className="w-full max-w-md mx-auto lg:mx-0">
              <CardContent className="p-3 sm:p-4">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,text/csv"
                  onChange={handleFileInputChange}
                  className="hidden"
                />

                {/* File Information Display */}
                {csvFileInfo ? (
                  <div className="py-1">
                    <div className="flex items-center justify-center gap-4 mb-1">
                      <span className="text-sm font-normal text-gray-800 font-sans leading-tight">{csvFileInfo.name}</span>
                      <span className="text-sm font-normal text-gray-500 font-sans leading-tight">
                        {csvFileInfo.uploadTime}
                      </span>
                      <span className="text-sm font-normal text-gray-700 font-sans leading-tight">
                        {csvFileInfo.data.length} records loaded
                      </span>
                    </div>
                    <div className="flex justify-center gap-2">
                      <Button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-1 px-4 py-3 text-sm font-sans touch-manipulation min-h-[44px]"
                      >
                        <Upload className="w-3 h-3" />
                        {isUploading ? 'Uploading...' : 'Select File'}
                      </Button>
                      <Button
                        onClick={() => {
                          localStorage.removeItem('csvFileInfo');
                          setCsvFileInfo(null);
                          setCsvData([]);
                        }}
                        variant="outline"
                        size="sm"
                        className="px-4 py-3 text-gray-600 hover:text-gray-800 hover:bg-gray-100 text-sm font-sans touch-manipulation min-h-[44px]"
                      >
                        Clear Cache
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center py-3">
                    <Button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isUploading}
                      size="sm"
                      className="flex items-center gap-2 text-sm font-sans px-6 py-3 touch-manipulation min-h-[44px]"
                    >
                      <Upload className="w-4 h-4" />
                      {isUploading ? 'Uploading...' : 'Select CSV File'}
                    </Button>
                  </div>
                )}

                {/* 兼容旧版本显示 */}
                {csvData.length > 0 && !csvFileInfo && (
                  <div className="mt-3">
                    <div className="text-center">
                      <span className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded">
                        ✓ {csvData.length} records loaded
                      </span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* 右侧：摄像头扫描模块 */}
          <div>
            {/* 模块2: 摄像头扫描 - 移动端优化 */}
            <Card className="w-full max-w-md mx-auto lg:mx-0">
              <CardContent className="p-3 sm:p-6">
                {!isCameraActive ? (
                  <div className="text-center py-6 sm:py-8">
                    <Camera className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-3 sm:mb-4 text-gray-400" />
                    <p className="text-sm sm:text-lg mb-3 sm:mb-4 px-2">Click to start camera scanning</p>
                    <Button 
                      onClick={startCamera} 
                      className="w-full sm:w-auto px-6 py-3 text-base touch-manipulation"
                      size="lg"
                    >
                      <Eye className="w-4 h-4 mr-2" />
                      Start Camera
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
                        className="w-full h-auto mx-auto rounded-lg border-2 border-gray-200 bg-black object-cover"
                        style={{ aspectRatio: '4/3', minHeight: '180px', maxHeight: '300px' }}
                      />
                      <canvas
                        ref={canvasRef}
                        className="hidden"
                      />

                      {/* OCR识别区域限定框 */}
                      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <div className="relative">
                          {/* 半透明遮罩 */}
                          <div className="absolute inset-0 bg-black bg-opacity-30 rounded-lg"></div>
                          
                          {/* 邮箱号码显示框 - 在识别框上方 */}
                          <div 
                            className="absolute bg-white text-red-600 font-bold rounded-md text-xs flex items-center justify-center font-mono overflow-hidden border-2 border-black"
                            style={{
                              width: 'min(300px, 80vw)',   // 与下方黑框相同宽度
                              height: '60px',   // 与下方黑框相同高度
                              top: '-80px',     // 在识别框上方
                              left: '-66px',    // 居中显示
                              fontSize: '20px', // 与下方黑框相同字体大小
                              lineHeight: '24px' // 相应调整行高
                            }}
                          >
                            {matchResults.length > 0 ? extractPOBoxNumber(matchResults[0].row.email || matchResults[0].row['PO Box'] || matchResults[0].row['po box'] || matchResults[0].row.pobox || '') || 'No Match' : 'No Match'}
                          </div>
                          
                          {/* 识别框 - 移动端自适应 */}
                          <div 
                            className="relative bg-transparent border-2 border-red-500 rounded-md shadow-lg"
                            style={{
                              width: 'min(168px, 70vw)',   // 移动端自适应宽度
                              height: '28px',   // 单行文字的高度
                              boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.3)' // 外部遮罩效果
                            }}
                          >
                          </div>
                          
                          {/* 识别文本显示框 - 透明背景，黑色边框和加粗字体 */}
                          <div 
                            className="absolute bg-transparent text-black font-bold rounded-md text-xs flex items-center justify-center font-mono overflow-hidden border-2 border-black"
                            style={{
                              width: 'min(300px, 80vw)',   // 适合摄像头框内的宽度
                              height: '60px',   // 高度增加50% (40px -> 60px)
                              top: '50px',      // 往上移动，更靠近识别框
                              left: '-66px',    // 居中显示
                              fontSize: '20px', // 字体大小进一步增加
                              lineHeight: '24px' // 相应调整行高
                            }}
                          >
                            {ocrText ? ocrText.substring(0, 25) : 'Waiting for recognition...'}
                          </div>
                        </div>
                      </div>

                      {isCameraActive && (
                        <div className="absolute top-2 left-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1 z-10">
                          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                          Recording
                        </div>
                      )}
                      {realTimeOCR && (
                        <div className="absolute top-2 right-2 bg-blue-500 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1 z-10">
                          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                          Real-time
                        </div>
                      )}
                      {!videoReady && isCameraActive && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg z-20">
                          <div className="text-white text-sm">Loading camera...</div>
                        </div>
                      )}
                    </div>
                    
                    {/* 移动端按钮布局优化 - 重新布局 */}
                    <div className="flex flex-col gap-3 px-2">
                      {/* 第一行：Real-time 和 Close 按钮 */}
                      <div className="flex justify-center gap-2">
                        <Button
                          onClick={toggleRealTimeOCR}
                          disabled={!videoReady}
                          variant={realTimeOCR ? "default" : "outline"}
                          size="sm"
                          className="text-xs px-4 py-3 flex-1 sm:flex-none min-w-0 touch-manipulation min-h-[44px]"
                        >
                          {realTimeOCR ? (
                            <>
                              <Square className="w-3 h-3 sm:mr-1" />
                              <span className="hidden sm:inline">Stop Recognition</span>
                              <span className="sm:hidden">Stop</span>
                            </>
                          ) : (
                            <>
                              <Play className="w-3 h-3 sm:mr-1" />
                              <span className="hidden sm:inline">Real-time</span>
                              <span className="sm:hidden">Real-time</span>
                            </>
                          )}
                        </Button>
                        <Button 
                          onClick={stopCamera} 
                          variant="outline" 
                          size="sm"
                          className="text-xs px-4 py-3 flex-1 sm:flex-none min-w-0 touch-manipulation min-h-[44px]"
                        >
                          <EyeOff className="w-3 h-3 sm:mr-1" />
                          <span className="hidden sm:inline">Close</span>
                          <span className="sm:hidden">Close</span>
                        </Button>
                      </div>
                      
                      {/* 第二行：Manual Recognition 按钮（更大） */}
                      <div className="flex justify-center">
                        <Button
                          onClick={() => captureAndProcessOCR(false)}
                          disabled={isProcessingOCR || realTimeOCR}
                          variant="default"
                          size="lg"
                          className="text-sm px-8 py-4 touch-manipulation min-h-[52px] font-semibold"
                        >
                          <Play className="w-4 h-4 mr-2" />
                          Manual Recognition
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>


        {/* 下方：匹配结果模块 - 移动端优化 */}
        <div className="flex justify-center">
          <Card className="w-full max-w-md mx-auto lg:mx-0">
            {matchResults.length > 0 && (
              <div className="p-3 sm:p-4 border-b border-gray-200">
                <div className="text-center">
                  <div className="text-sm sm:text-base text-gray-800">
                    <div className="flex justify-center gap-2 flex-wrap">
                      <span className="text-red-600 font-bold">{extractPOBoxNumber(matchResults[0].row.email || matchResults[0].row['PO Box'] || matchResults[0].row['po box'] || matchResults[0].row.pobox || '') || '—'}</span>
                      <span>{matchResults[0].row.name || '—'}</span>
                      <span>{(() => {
                        const streetNum = matchResults[0].row.streetNumber;
                        const address = matchResults[0].row.address;
                        console.log('显示地址 - streetNumber:', JSON.stringify(streetNum), 'address:', JSON.stringify(address));
                        return (streetNum ? streetNum + ' ' : '') + (address || '—');
                      })()}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <CardContent className="p-3 sm:p-6">
              {matchResults.length === 0 ? (
                <div className="text-center py-6 sm:py-8 text-gray-500">
                  <p className="text-sm sm:text-base">No matching results</p>
                </div>
              ) : (
                <div className="text-center text-xs sm:text-sm text-gray-600 bg-gray-100 p-2 sm:p-3 rounded border border-gray-200">
                  Found {matchResults.length} matching records
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